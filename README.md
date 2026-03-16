# Orquestador Multi-Agentes

Sistema de respuesta a consultas internas que recibe una pregunta del usuario, la clasifica y la rutea a uno o varios agentes especializados. Cada agente cuenta con su propio RAG (Retrieval-Augmented Generation) construido sobre documentación de procedimientos del área. El flujo es orquestado con **LangGraph**, implementado con **LangChain** y trazado con **Langfuse**.

## ¿Cómo funciona?

1. El usuario ingresa una consulta por consola.
2. Un clasificador LLM detecta a qué dominio/s pertenece la pregunta: `hr`, `finance`, `legal` o `tech`.
3. Se despachan los agentes correspondientes en paralelo (patrón **orchestrator-worker** via `Send()` de LangGraph).
4. Cada agente busca en su base vectorial (ChromaDB) y genera una respuesta usando el contexto recuperado.
5. Si hay múltiples agentes, un nodo sintetizador unifica las respuestas.
6. La respuesta final es evaluada (relevancia, completitud, precisión) y los scores se registran en Langfuse.
7. El resultado completo se guarda como JSON en `output/`.

## Estructura del proyecto

```
orquestador-multi-agentes/
│
├── data/
│   ├── finance_docs.txt        # Documentación de procedimientos de Finanzas
│   ├── hr_docs.txt             # Documentación de procedimientos de RRHH
│   ├── legal_docs.txt          # Documentación de procedimientos Legales
│   └── tech_docs.txt           # Documentación de procedimientos de IT
│
├── output/                     # Respuestas generadas guardadas como JSON
│
└── src/
    ├── agents/                 # Instancias de los 4 agentes especializados
    │   ├── finance_agent.py
    │   ├── hr_agent.py
    │   ├── legal_agent.py
    │   └── tech_agent.py
    │
    ├── rag/
    │   └── domain_rag.py       # Clase base: carga docs, crea chunks, embeddings y retrievers
    │
    ├── orchestrator.py         # Grafo LangGraph: clasificación, workers y síntesis
    ├── evaluator.py            # Evaluación de respuestas y envío de scores a Langfuse
    ├── output_writer.py        # Serialización de resultados a JSON
    ├── exceptions.py           # Excepciones personalizadas
    └── logger.py               # Logger con colores y formato personalizado
```

## Requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) para manejo de dependencias

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/iv-chucoff/orquestador-multi-agentes.git
cd orquestador-multi-agentes
```

### 2. Crear el entorno virtual

```bash
uv venv
```

### 3. Activar el entorno

```powershell
.venv\Scripts\activate.ps1
```

### 4. Instalar dependencias

```bash
uv sync
```

### 5. Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
OPENAI_API_KEY=tu-api-key-aqui
MODEL_NAME=gpt-4o-mini

LANGFUSE_SECRET_KEY=tu-secret-key
LANGFUSE_PUBLIC_KEY=tu-public-key
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

- Obtener API Key de OpenAI: [platform.openai.com](https://platform.openai.com)
- Obtener credenciales de Langfuse: [cloud.langfuse.com](https://cloud.langfuse.com)

## Ejecución

```bash
python -m src.orchestrator
```

Escribir la consulta cuando se solicite. Para salir ingresar `exit`, `quit` o `salir`.

## Ejemplos

### Consulta a un único dominio

**Pregunta:** `¿Puedo solicitar un adelanto de sueldo?`

```json
{
  "timestamp": "20260316_192844_049880",
  "query": "¿Puedo solicitar un adelanto de sueldo?",
  "domains": ["finance"],
  "latency_seconds": 2.969,
  "scores": {
    "relevance": 1.0,
    "completeness": 1.0,
    "accuracy": 1.0
  },
  "final_answer": "SolvNet no otorga adelantos de sueldo como política general. En situaciones de emergencia debidamente justificadas, el empleado puede solicitar una excepción al CFO a través de HR."
}
```

---

### Consulta a múltiples dominios

**Pregunta:** `¿Cómo solicito días de vacaciones? ¿Cómo restauro mi contraseña?`

La pregunta involucra dos dominios distintos (`hr` y `tech`), por lo que ambos agentes son despachados en paralelo. Cada uno busca en su propia base de conocimiento y un nodo sintetizador unifica la respuesta final.

```json
{
  "timestamp": "20260316_194005_078913",
  "query": "Como solicito dias de vacaciones? como restauro mi contraseña?",
  "domains": ["hr", "tech"],
  "latency_seconds": 4.764,
  "scores": {
    "relevance": 1.0,
    "completeness": 1.0,
    "accuracy": 1.0
  },
  "final_answer": "Para solicitar días de vacaciones, debes ingresar al portal de autoservicio de HRMS. Allí, selecciona \"Solicitud de licencia\", elige el tipo de licencia y las fechas deseadas. La solicitud se enviará automáticamente a tu responsable para su aprobación.\n\nEn cuanto a la restauración de tu contraseña, se recomienda utilizar 1Password, que es la herramienta corporativa para la gestión de contraseñas. Si necesitas asistencia adicional, puedes abrir un ticket de solicitud de software o consultar con tu responsable."
}
```

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Orquestación | LangGraph |
| Framework LLM | LangChain |
| Modelos | OpenAI GPT-4o-mini |
| Base vectorial | ChromaDB |
| Trazabilidad | Langfuse |
| Gestión de dependencias | uv |

## Contacto

**Autor:** Ivana Chucoff
**Proyecto:** orquestador-multi-agentes
**Versión:** 0.1.0
