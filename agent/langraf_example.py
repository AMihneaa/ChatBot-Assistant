
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

LM_BASE_URL = "http://127.0.0.1:1234/v1"
LM_API_KEY  = "lm-studio"
# MODEL_NAME  = "jan-v1-4b"
MODEL_NAME = "qwen/qwen3-vl-4b"

DOCS_DIR  = "./docs"
INDEX_DIR = "./faiss_index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

llm = ChatOpenAI(
    model=MODEL_NAME, base_url=LM_BASE_URL, api_key=LM_API_KEY,
    temperature=0.3, max_tokens=1024,
)

embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

def build_or_load_index():
    Path(DOCS_DIR).mkdir(exist_ok=True, parents=True)
    if Path(INDEX_DIR).exists():
        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    urls = ["https://docs.langchain.com/oss/python/langgraph/agentic-rag"]
    docs = WebBaseLoader(urls).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

vs = build_or_load_index()

def answer_question(query: str) -> str:
    docs = vs.similarity_search(query, k=4)

    context = "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    system = (
        "Ești un asistent care răspunde concis pe baza contextului furnizat.\n"
        "- Folosește DOAR informațiile din context pentru răspuns.\n"
        "- Dacă informația nu e în context, spune clar că nu este prezentă și evită halucinațiile.\n"
        "- Citează fragmentele relevante folosind indexul lor (ex. [1], [2], ...)."
    )

    user = (
        f"Întrebare: {query}\n\n"
        f"Context (fragmente):\n{context}\n\n"
        "Instrucțiuni: Citează fragmentele relevante (ex. [1], [2]...) și răspunde în română."
    )

    res = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user} /no_think"},
    ])

    print("\n[----- DEBUG] Primul chunk:\n", docs[0].page_content[:500], " [Debug -----]\n")

    return res.content


if __name__ == "__main__":
    while True:
        q = input("\nQ> ").strip()
        if not q: break
        print("\nA>", answer_question(q))
