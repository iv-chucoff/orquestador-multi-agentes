"""Excepciones personalizadas para el orquestador multi-agente RAG."""


class OrchestratorError(Exception):
    """Excepción base del sistema. Captura cualquier error con un solo except."""
    pass


class APIError(OrchestratorError):
    """Error de APIs externas (OpenAI, Langfuse): clave inválida, fallo de conexión, etc."""
    pass


class InputError(OrchestratorError):
    """Consulta del usuario vacía o inválida."""
    pass


class DocumentError(OrchestratorError):
    """Archivo de documentos no encontrado o no se puede leer."""
    pass


class VectorStoreError(OrchestratorError):
    """Fallo al crear, cargar o consultar el vectorstore de ChromaDB."""
    pass


class ClassificationError(OrchestratorError):
    """El clasificador no encontró ningún dominio válido para la consulta."""
    pass


class AgentError(OrchestratorError):
    """Un agente de dominio falló durante su ejecución."""

    def __init__(self, domain: str, message: str) -> None:
        self.domain = domain
        super().__init__(f"[{domain.upper()}] {message}")


class EvaluationError(OrchestratorError):
    """Fallo al evaluar o puntuar una respuesta en Langfuse."""
    pass
