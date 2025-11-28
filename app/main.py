from fastapi import FastAPI
from .api.routes import router as chat_router


def create_app() -> FastAPI:
    app = FastAPI(title="Ticket Assistant API")

    app.include_router(chat_router, prefix="", tags=["chat"])

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    print("Pornesc API-ul LLM pe http://localhost:8001 ...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
