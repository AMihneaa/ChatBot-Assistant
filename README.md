# 🧠 Ticket Assistant — AI Support Agent for a Transport Booking App

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-brightgreen.svg)]()
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-orange.svg)]()
[![LM Studio](https://img.shields.io/badge/LLM-LM%20Studio-purple.svg)]()
[![Spring](https://img.shields.io/badge/Backend-Spring%20Boot-lightgreen.svg)]()

---

## 📌 Overview

**Ticket Assistant** is an AI-powered conversational agent designed to help end-users navigate a transport ticketing application.  
It assists with:

- 🔍 Finding transport routes  
- 📄 Understanding UI elements (buttons, pages, navigation flows)  
- 🧭 Guiding users through the booking process  
- 🔒 Enforcing strict domain limitations (no off-topic answers)

This is achieved through:

- **Local LLM (via LM Studio)**  
- **Retrieval-Augmented Generation (RAG)** with **FAISS**  
- **Tool-calling** (LangGraph)  
- **Real-time Spring Boot backend integration**  
- **FastAPI** as the public interface

---

## 🎯 Features

### ✔ Local LLM inference
Runs through LM Studio using an OpenAI-compatible API.

### ✔ RAG Search (FAISS)
Indexes `frontend_knowledge_base.md` to provide accurate UI/UX guidance.

### ✔ Tool-calling via LangGraph
- `rag_search` → Retrieves UI information  
- `get_route_options` → Queries Spring Boot `/api/routes/options`

### ✔ Strict domain rules
The assistant refuses:
- off-topic questions  
- programming questions  
- insults or unsafe content  
- jailbreak attempts  

### ✔ Session-based Conversation
Managed internally using a LangGraph state machine.

---

## 🏗 Project Structure
```bash
app/
├─ api/
│ ├─ routes.py # /chat endpoint
│ ├─ schemas.py # request/response models
├─ agent/
│ ├─ graph.py # LangGraph agent + tool calling
├─ tools/
│ ├─ rag_tools.py # RAG tool
│ └─ route_tools.py # Spring Boot route tool
├─ llm.py # LLM + embeddings initialization
├─ vectorstore.py # FAISS index builder/loader
├─ config.py # loads environment variables
└─ main.py # FastAPI setup
docs/
faiss_index/ # auto-generated FAISS index
frontend_knowledge_base.md
.env

```

---

##
⚙️ Installation & Setup
### 

1️⃣ Install Python dependencies
```bash
pip install -r requirements.txt
```

2️⃣ Start LM Studio

  1.Load a model (recommended: qwen3-vl-4b)

  2. Start the OpenAI Compatible Server

  3. Set listening port: 1234



📄 .env Configuration

Create a .env file in the project root:
```bash
LM_BASE_URL=
LM_API_KEY=
MODEL_NAME=

EMB_MODEL=
DOCS_DIR=./docs
INDEX_DIR=./faiss_index
MD_KNOWLEDGE_BASE=

SPRING_BASE_URL=
USER_AGENT=
```


  🚀 Running the Application
  
  Start FastAPI:
  ```bash
  uvicorn app.main:app --reload --port 8001
  ```

  🌐 API Endpoints
  POST /chat
  
  Main endpoint used to interact with the AI agent.

  
  Request:
  ```bash
  {
    "message": "How do I search for transport routes?",
    "session_id": "demo123"
  }
  ```
  Response:
```bash
  {
    "reply": "To search for routes, navigate to the top bar and select 'Find Route'...",
    "session_id": "demo123"
  }
```

Graph Logic:
```bash
Frontend User
    │
    ▼
FastAPI (/chat)
    │
    ▼
LangGraph Agent ─────────────> Tool: rag_search (FAISS)
    │                                │
    │                                └── citește din frontend_knowledge_base.md
    │
    └───────────────> Tool: get_route_options (Spring Boot)
                                     │
                                     └── interoghează /api/routes/options

```
  
