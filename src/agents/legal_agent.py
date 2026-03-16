from src.exceptions import AgentError
from src.rag.domain_rag import DomainRAG

rag = DomainRAG(domain="legal", doc_path="data/legal_docs.txt")
rag.build()


def run(query: str) -> dict:
    try:
        return rag.answer_with_metadata(query)
    except Exception as e:
        raise AgentError("legal", str(e)) from e
