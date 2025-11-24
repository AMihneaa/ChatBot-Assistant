from pathlib import Path
import os

from dotenv import load_dotenv
import httpx

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# ===================== Config & ENV =====================

load_dotenv()

os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "m-rag-agent/0.1"))

LM_BASE_URL = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234/v1")
LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-vl-4b")

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
INDEX_DIR = os.getenv("INDEX_DIR", "./faiss_index")

SPRING_BASE_URL = os.getenv("SPRING_BASE_URL", "http://localhost:8080")


# ===================== LLM & Embeddings =====================

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=LM_BASE_URL,
    api_key=LM_API_KEY,
    temperature=0.3,
    max_tokens=1024,
)

embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)


# ===================== FAISS: build / load index =====================

def build_or_load_index():
    Path(DOCS_DIR).mkdir(exist_ok=True, parents=True)
    index_path = Path(INDEX_DIR)

    if index_path.exists():
        print(f"[RAG] Încarc indexul FAISS din {INDEX_DIR}")
        return FAISS.load_local(
            folder_path=INDEX_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    print("[RAG] Nu există index. Construiesc unul nou...")

    urls = ["https://docs.langchain.com/oss/python/langgraph/agentic-rag"]
    docs = WebBaseLoader(urls).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    vs_ = FAISS.from_documents(chunks, embeddings)
    vs_.save_local(INDEX_DIR)
    print(f"[RAG] Am salvat indexul FAISS în {INDEX_DIR}")
    return vs_


vs = build_or_load_index()


# ===================== TOOLS =====================

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


@tool
def get_route_options(departure: str, arrival: str) -> str:
    """
    Caută opțiuni de rute (tren / bus / avion) între două locații,
    folosind backend-ul Spring Boot.

    - `departure`: punct de plecare (ex: 'Bucuresti')
    - `arrival`: destinație (ex: 'Constanta')

    Returnează un text rezumat cu cele mai relevante opțiuni
    sau un mesaj de eroare foarte explicit.
    """
    url = f"{SPRING_BASE_URL}/api/routes/options"
    print(f"[TOOL] get_route_options called with departure={departure!r}, arrival={arrival!r}")
    print(f"[TOOL] HTTP GET {url}")

    try:
        resp = httpx.get(
            url,
            params={"departure": departure, "arrival": arrival},
            timeout=10.0,
        )
    except Exception as e:
        msg = f"[EROARE_TOOL] Nu am putut face request la {url}: {type(e).__name__}: {e!r}"
        print(msg)
        return msg

    print(f"[TOOL] Server response status: {resp.status_code}")
    print(f"[TOOL] Server raw body: {resp.text!r}")

    if resp.status_code != 200:
        msg = (
            f"[EROARE_TOOL] Serverul Spring a răspuns cu status {resp.status_code} "
            f"pentru {departure} → {arrival}. Body: {resp.text}"
        )
        print(msg)
        return msg

    try:
        data = resp.json()
    except Exception as e:
        msg = (
            f"[EROARE_TOOL] Răspunsul de la server nu este JSON valid: {type(e).__name__}: {e!r}. "
            f"Body brut: {resp.text}"
        )
        print(msg)
        return msg

    if not data:
        msg = (
            f"Nu am găsit rute pentru {departure} → {arrival} "
        )
        print(f"[TOOL] {msg}")
        return msg

    lines = [f"Am găsit {len(data)} opțiuni de rute pentru {departure} → {arrival}:\n"]

    for i, option in enumerate(data[:5], start=1):
        legs = option.get("legs", [])
        if not legs:
            continue

        lines.append(f"Opțiunea {i}:")
        for j, leg in enumerate(legs, start=1):
            r_type = leg.get("transportType", "UNKNOWN")
            r_id = leg.get("routeId", "?")
            t_id = leg.get("transportId", "?")
            stops = leg.get("stops", [])
            seats = leg.get("availableSeats", "?")

            route_path = " → ".join(stops) if stops else "N/A"
            lines.append(
                f"  Segment {j}: [{r_type}] {t_id} (routeId={r_id})\n"
                f"    Traseu: {route_path}\n"
                f"    Locuri disponibile: {seats}"
            )

        lines.append("")

    result = "\n".join(lines)
    print(f"[TOOL] get_route_options result:\n{result}")
    return result



tools = [rag_search, get_route_options]
model_with_tools = llm.bind_tools(tools)


# ===================== LangGraph: noduri & graf =====================

def call_model(state: MessagesState):
    """
    Nodul principal: cheamă LLM-ul cu tool calling activ.
    """
    system_prompt = (
        "Ești un asistent de suport pentru aplicația de achiziție bilete.\n"
        "- Dacă utilizatorul întreabă despre structura site-ului, pagini, meniuri, "
        "folosește tool-ul `rag_search` pentru a căuta în documentație și explică pașii.\n"
        "- Dacă utilizatorul întreabă despre rute, transport, opțiuni de călătorie "
        "(ex: 'ce rute am din București în Cluj?'), folosește tool-ul `get_route_options`.\n"
        "- Răspunde mereu în română, clar, cu explicații pas cu pas dacă este nevoie.\n"
        "- Dacă nu ai informațiile necesare, explică ce lipsește în loc să inventezi."
    )

    messages = state["messages"]
    full_messages = [SystemMessage(content=system_prompt), *messages]

    response = model_with_tools.invoke(full_messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

graph = StateGraph(MessagesState)

graph.add_node("model", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("model")

graph.add_conditional_edges(
    "model",
    tools_condition,
    {
        "tools": "tools",     
        "__end__": "__end__", 
    },
)

graph.add_edge("tools", "model")  


app = graph.compile()


# ===================== CLI de test =====================

if __name__ == "__main__":
    print("Asistentul de suport a pornit. Întreabă ceva (Enter gol = exit).")
    state = {"messages": []}

    while True:
        q = input("\nQ> ").strip()
        if not q:
            break

        state["messages"].append(HumanMessage(content=q))

        result = app.invoke(state)
        state = result

        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print("\nA>", last_msg.content)
        else:
            print("\nA>", last_msg)
