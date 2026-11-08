import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "Research_agent.backend.api.app:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
    )
