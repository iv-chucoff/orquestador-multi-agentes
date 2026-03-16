from langfuse import Langfuse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.exceptions import EvaluationError
from src.logger import get_logger

logger = get_logger("evaluator")

langfuse_client = Langfuse()


class EvaluationScores(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0, description="¿La respuesta responde la pregunta?")
    completeness: float = Field(ge=0.0, le=1.0, description="¿La respuesta está completa?")
    accuracy: float = Field(ge=0.0, le=1.0, description="¿La información parece correcta?")
    reasoning: str = Field(description="Breve explicación de los puntajes")


def evaluate_response(trace_id: str, query: str, answer: str) -> EvaluationScores | None:
    """Puntúa la respuesta en 3 dimensiones, postea los scores al trace de Langfuse y los retorna."""
    logger.info("Evaluando respuesta para trace %s...", trace_id)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Eres un evaluador de calidad de un sistema RAG multi-agente corporativo. "
                "Analizá la respuesta generada y puntuá cada dimensión entre 0.0 y 1.0:\n"
                "- relevance: ¿La respuesta responde directamente la pregunta del usuario?\n"
                "- completeness: ¿La respuesta cubre todos los aspectos de la pregunta o le falta información clave?\n"
                "- accuracy: ¿La información parece correcta, consistente y sin contradicciones?\n\n"
                "Sé estricto: un 1.0 implica una respuesta perfecta en esa dimensión.",
            ),
            ("human", "Pregunta: {query}\n\nRespuesta: {answer}"),
        ])
        chain = prompt | llm.with_structured_output(EvaluationScores)
        scores: EvaluationScores = chain.invoke({"query": query, "answer": answer})

        for metric in ["relevance", "completeness", "accuracy"]:
            langfuse_client.create_score(
                trace_id=trace_id,
                name=metric,
                value=getattr(scores, metric),
                comment=scores.reasoning,
            )
        logger.info(
            "Scores posteados: relevance=%.2f, completeness=%.2f, accuracy=%.2f",
            scores.relevance, scores.completeness, scores.accuracy,
        )
        return scores
    except Exception as e:
        raise EvaluationError(f"No se pudo evaluar el trace {trace_id}: {e}") from e
