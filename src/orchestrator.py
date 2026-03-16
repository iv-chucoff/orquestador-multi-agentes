import operator
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from src.agents import finance_agent, hr_agent, legal_agent, tech_agent
from src.evaluator import evaluate_response
from src.exceptions import ClassificationError
from src.logger import get_logger
from src.output_writer import save_output

load_dotenv()

logger = get_logger("orchestrator")


# ------------------------------------------------------------------
# Estado del grafo
# ------------------------------------------------------------------

class OrchestratorState(TypedDict):
    query: str
    domains: list[str]
    domain_queries: list[dict]
    agent_outputs: Annotated[list[dict], operator.add]
    final_answer: str


# ------------------------------------------------------------------
# Clasificación de dominio
# ------------------------------------------------------------------

VALID_DOMAINS = ["hr", "finance", "legal", "tech"]


class DomainQuery(BaseModel):
    domain: str
    sub_query: str


class Classification(BaseModel):
    items: list[DomainQuery]


def classify(state: OrchestratorState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """Eres un clasificador de consultas de una empresa. Dado un mensaje, identifica a qué dominios pertenece
        y extrae la sub-consulta específica para cada uno.

        Dominios disponibles: {VALID_DOMAINS}.

        Descripción detallada de cada dominio y sus temas cubiertos:

        - hr: Todo lo relacionado con personas y empleo interno.
        Temas: incorporación de empleados (onboarding), licencias y ausencias, evaluación de desempeño,
        compensaciones y beneficios, desvinculaciones y offboarding, diversidad e inclusión,
        capacitación y desarrollo profesional, clima laboral y bienestar, trabajo remoto e híbrido,
        código de ética y conducta, relaciones sindicales y gremiales, documentación de empleados,
        pasantes y programas universitarios, métricas de HR.

        - finance: Todo lo relacionado con dinero, contabilidad y operaciones financieras.
        Temas: gestión de gastos y reembolsos, facturación y cuentas por cobrar, presupuesto y
        planificación financiera, nómina y liquidación de haberes, compras y proveedores, impuestos
        y cumplimiento fiscal, caja chica y fondos fijos, inversiones y tesorería, auditoría interna,
        financiamiento bancario, cierre contable mensual y anual, política de viajes corporativos,
        indicadores y reportes financieros.

        - legal: Todo lo relacionado con contratos, normativa y cumplimiento jurídico.
        Temas: contratos con clientes, protección de datos y privacidad, propiedad intelectual,
        contratos laborales, cumplimiento regulatorio y gobierno corporativo, litigios y disputas,
        licencias de software y open source (aspecto legal/contractual), acuerdos con socios y alianzas,
        normativa internacional, seguros corporativos, política antitrust, gestión documental legal,
        política de regalos corporativos, métricas del área legal.

        - tech: Todo lo relacionado con sistemas, infraestructura y soporte tecnológico.
        Temas: mesa de ayuda y tickets de soporte, gestión de accesos y permisos, equipamiento y
        hardware, seguridad informática, infraestructura y nube, software autorizado (aspecto técnico/
        operativo), redes y conectividad, copias de seguridad, activos tecnológicos, soporte a
        aplicaciones de negocio, continuidad del negocio y recuperación ante desastres,
        comunicaciones y videoconferencias, monitoreo de sistemas, proyectos de IT y gestión de cambios.

        REGLAS PARA CASOS AMBIGUOS:
        - "Licencias de software": si la pregunta es sobre CONTRATOS o COSTOS → legal o finance;
        si es sobre INSTALACIÓN, ACCESO o USO → tech.
        - "Nómina": si es sobre LIQUIDACIÓN o PAGOS → finance; si es sobre ALTAS/BAJAS de empleados → hr.
        - "Viajes": si es sobre REEMBOLSOS o PRESUPUESTO → finance; si es sobre POLÍTICAS para empleados → hr.
        - "Contratos laborales": si es sobre CLÁUSULAS LEGALES → legal; si es sobre CONDICIONES DE EMPLEO → hr.
        - Una consulta puede pertenecer a MÚLTIPLES dominios si genuinamente involucra ambos.

        Devuelve solo los dominios relevantes con su sub-consulta correspondiente."""
        ),
        ("human", "{query}"),
    ])
    chain = prompt | llm.with_structured_output(Classification)
    result = chain.invoke({"query": state["query"], "VALID_DOMAINS": VALID_DOMAINS})
    valid = [item.model_dump() for item in result.items if item.domain in VALID_DOMAINS]
    if not valid:
        raise ClassificationError(f"No se encontró ningún dominio válido para: '{state['query']}'")
    domains = [item["domain"] for item in valid]
    logger.info("Dominios detectados: %s", domains)
    return {"domain_queries": valid, "domains": domains}


# ------------------------------------------------------------------
# Dispatch: lanza un worker por dominio
# ------------------------------------------------------------------

def dispatch(state: OrchestratorState) -> list[Send]:
    return [
        Send(f"{dq['domain']}_agent", {**state, "sub_query": dq["sub_query"]})
        for dq in state["domain_queries"]
    ]


# ------------------------------------------------------------------
# Nodos worker (uno por agente)
# ------------------------------------------------------------------

def hr_agent_node(state: OrchestratorState) -> dict:
    result = hr_agent.run(state["sub_query"])
    return {"agent_outputs": [{"domain": "hr", **result}]}


def finance_agent_node(state: OrchestratorState) -> dict:
    result = finance_agent.run(state["sub_query"])
    return {"agent_outputs": [{"domain": "finance", **result}]}


def legal_agent_node(state: OrchestratorState) -> dict:
    result = legal_agent.run(state["sub_query"])
    return {"agent_outputs": [{"domain": "legal", **result}]}


def tech_agent_node(state: OrchestratorState) -> dict:
    result = tech_agent.run(state["sub_query"])
    return {"agent_outputs": [{"domain": "tech", **result}]}


# ------------------------------------------------------------------
# Finalize: pasa directo si es 1 dominio, sintetiza si son varios
# ------------------------------------------------------------------

def finalize(state: OrchestratorState) -> dict:
    outputs = state["agent_outputs"]

    if len(outputs) == 1:
        return {"final_answer": outputs[0]["answer"]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un agente sintetizador. Recibiste respuestas de múltiples agentes especializados. "
            "Combiná la información en una única respuesta clara y coherente, sin repetir contenido.",
        ),
        (
            "human",
            "Consulta original: {query}\n\n"
            "Respuestas de los agentes:\n{agent_outputs}",
        ),
    ])
    formatted = "\n\n".join(
        f"[{o['domain'].upper()}]: {o['answer']}" for o in outputs
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "agent_outputs": formatted})
    return {"final_answer": response.content}


# ------------------------------------------------------------------
# Construcción del grafo
# ------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify)
    graph.add_node("hr_agent", hr_agent_node)
    graph.add_node("finance_agent", finance_agent_node)
    graph.add_node("legal_agent", legal_agent_node)
    graph.add_node("tech_agent", tech_agent_node)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", dispatch, ["hr_agent", "finance_agent", "legal_agent", "tech_agent"])
    graph.add_edge("hr_agent", "finalize")
    graph.add_edge("finance_agent", "finalize")
    graph.add_edge("legal_agent", "finalize")
    graph.add_edge("tech_agent", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------
# Punto de entrada
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Construimos el grafo UNA sola vez
    app = build_graph()
    # Un id fijo para esta sesión de chat en terminal
    thread_id = "cli_chat"
    print("Chat multi-agente. Escribe 'salir' para terminar.\n")
    while True:
        query = input("Tú: ").strip()
        if query.lower() in {"salir", "exit", "quit"}:
            break
        if not query:
            print("Por favor ingresá tu consulta.\n")
            continue

        # Un handler por request: permite recuperar el trace_id de esta invocación
        handler = CallbackHandler()
        start_time = time.time()
        try:
            result = app.invoke(
                {"query": query, "agent_outputs": []},
                config={
                    "configurable": {"thread_id": thread_id},
                    "callbacks": [handler],
                },
            )
        except ClassificationError:
            print("Bot: No pude identificar el tema de tu consulta. Por favor ingresá una pregunta sobre Recursos Humanos, Finanzas, Legal o Tecnología.\n")
            continue
        latency = time.time() - start_time
        print(f"Bot: {result['final_answer']}\n")

        # Evaluación: puntúa en Langfuse y retorna los scores para el JSON
        scores = None
        trace_id = handler.last_trace_id
        if trace_id:
            scores = evaluate_response(trace_id, query, result["final_answer"])

        save_output(query, result, latency, scores)
