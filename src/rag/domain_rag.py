from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from src.exceptions import DocumentError, VectorStoreError
from src.logger import get_logger

load_dotenv()

_DEFAULT_PERSIST_DIR = Path("data/chroma_db")


class DomainRAG:
    """RAG agent especializado en un dominio específico.

    Uso:
        rag = DomainRAG(domain="hr", doc_path="data/hr_docs.txt")
        rag.build()
        response = rag.answer("¿Cuántos días de vacaciones tengo?")
    """

    def __init__(
        self,
        domain: str,
        doc_path: str | Path,
        chunk_size: int = 600,
        chunk_overlap: int = 60,
        model: str = "gpt-4o-mini",
        k: int = 4,
        persist_directory: str | Path = _DEFAULT_PERSIST_DIR,
    ):
        self.domain = domain
        self.doc_path = Path(doc_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.persist_directory = Path(persist_directory)
        self._llm = ChatOpenAI(model=model, temperature=0)
        self._logger = get_logger(f"rag.{domain}")

    # ------------------------------------------------------------------
    # Pasos internos del pipeline
    # ------------------------------------------------------------------

    def _load_documents(self) -> list[Document]:
        if not self.doc_path.exists():
            raise DocumentError(f"Archivo no encontrado: {self.doc_path}")
        loader = TextLoader(str(self.doc_path), encoding="utf-8")
        return loader.load()

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def _build_vectorstore(self, chunks: list[Document]) -> Chroma:
        embeddings = OpenAIEmbeddings()
        collection_name = f"{self.domain}_collection"
        persist_dir = str(self.persist_directory)

        # Si ya existe la colección persistida, cargarla sin regenerar embeddings
        if self.persist_directory.exists():
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            if vectorstore._collection.count() > 0:
                # Verificar que el índice HNSW sea legible (puede estar corrupto)
                try:
                    vectorstore.similarity_search("test", k=1)
                    self._logger.info("Vectorstore '%s' cargado desde disco (%d docs)", collection_name, vectorstore._collection.count())
                    return vectorstore
                except Exception:
                    self._logger.warning("Vectorstore '%s' en disco está corrupto, reconstruyendo...", collection_name)
                    vectorstore.delete_collection()

        # Primera vez (o reconstrucción tras corrupción): generar embeddings y persistir en disco
        self._logger.info("Construyendo vectorstore '%s' desde cero (%d chunks)...", collection_name, len(chunks))
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )
        except Exception as e:
            raise VectorStoreError(f"No se pudo crear el vectorstore '{collection_name}': {e}") from e
        self._logger.info("Vectorstore '%s' creado y persistido.", collection_name)
        return vectorstore

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Lee el documento, genera chunks, embeddings y construye el vectorstore."""
        self._logger.info("Iniciando build para dominio '%s'...", self.domain)
        docs = self._load_documents()
        chunks = self._chunk_documents(docs)
        self._logger.info("%d chunks generados para '%s'.", len(chunks), self.domain)
        self.vectorstore = self._build_vectorstore(chunks)

    def retrieve(self, query: str) -> list[Document]:
        """Devuelve los k chunks más relevantes para la query."""
        try:
            return self.vectorstore.similarity_search(query, k=self.k)
        except Exception as e:
            raise VectorStoreError(f"Fallo en similarity_search para '{self.domain}': {e}") from e

    def _build_prompt_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Eres un agente especialista en {domain}. "
                "Responde basándote en el contexto provisto. "
                "Si el contexto tiene información relevante aunque sea general, úsala para responder. "
                "Solo di 'No tengo información sobre eso.' si el tema no aparece en absoluto en el contexto.\n\n"
                "Contexto:\n{context}",
            ),
            ("human", "{query}"),
        ])
        return prompt | self._llm

    def answer_with_metadata(self, query: str) -> dict:
        """Recupera contexto relevante, genera una respuesta y devuelve los chunks usados."""
        self._logger.info("Consulta recibida en '%s': %s", self.domain, query)
        docs = self.retrieve(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        chain = self._build_prompt_chain()
        response = chain.invoke({"domain": self.domain, "context": context, "query": query})
        self._logger.info("Respuesta generada para '%s'.", self.domain)
        return {
            "answer": response.content,
            "retrievers": [doc.page_content for doc in docs],
        }
