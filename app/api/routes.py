from fastapi import APIRouter

from .schemas import ChatRequest, ChatResponse
from ..agent.graph import run_turn

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Endpoint-ul pe care îl va chema clientul web.
    """
    reply, session_id = run_turn(req.message, req.session_id)
    return ChatResponse(reply=reply, session_id=session_id)
