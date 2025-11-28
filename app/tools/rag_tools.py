from langchain_core.tools import tool
from ..vectorstore import vs


@tool
def rag_search(query: str) -> str:
    """
    Caută în indexul FAISS fragmente relevante pentru întrebare
    și întoarce contextul concatenat, numerotat [1], [2], etc.
    Folosit pentru întrebări despre documentație / structura site-ului.
    """
    print(f"[TOOL] rag_search called with query: {query!r}")
    docs = vs.similarity_search(query, k=4)
    if not docs:
        return "Nu am găsit fragmente relevante în index."

    context = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )
    return context
