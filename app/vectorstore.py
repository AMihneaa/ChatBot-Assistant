from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .llm import embeddings
from .config import get_settings

settings = get_settings()


def build_or_load_index() -> FAISS:
    """
    Build or load the FAISS index based on `frontend_knowledge_base.md`
    generated from the frontend source code.
    """
    index_path: Path = settings.index_dir

    if index_path.exists():
        print(f"[RAG] Incarc indexul FAISS din {index_path}")
        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    md_path: Path = settings.md_knowledge_base
    if not md_path.exists():
        raise FileNotFoundError(
            "[RAG] Nu am gasit `frontend_knowledge_base.md` in directorul curent.\n"
        )

    print(f"[RAG] Construiesc index FAISS din {md_path} ...")

    text = md_path.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": str(md_path)})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    print(f"[RAG] Am impartit front-end-ul in {len(chunks)} chunk-uri.")

    vs_ = FAISS.from_documents(chunks, embeddings)
    vs_.save_local(str(index_path))

    print(f"[RAG] Am salvat indexul FAISS in {index_path}")
    return vs_


vs = build_or_load_index()
