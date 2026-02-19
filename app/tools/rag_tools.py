from langchain_core.tools import tool
from ..vectorstore import vs


@tool
def rag_search(query: str) -> str:
    """
    Cauta in indexul FAISS fragmente relevante pentru întrebare
    și intoarce contextul concatenat, numerotat [1], [2], etc.
    Folosit pentru întrebari despre documentație / structura site-ului.
    """
    print(f"[TOOL] rag_search called with query: {query!r}")
    docs = vs.similarity_search(query, k=4)
    if not docs:
        return "Nu am gasit fragmente relevante in index."

    context = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )
    return context
