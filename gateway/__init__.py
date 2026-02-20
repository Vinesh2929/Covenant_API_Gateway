"""
gateway package

Top-level package for the AI Gateway service. Exposes the FastAPI application
instance so that ASGI servers (uvicorn) and test clients can import it directly
via `from gateway import app`.
"""
