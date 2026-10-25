import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from doubtdesk.api.routes import router
from doubtdesk.graph.builder import build_graph
from doubtdesk.ingestion.embeddings import load_embedding_model
from doubtdesk.retrieval.vector_store import get_collection

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    app.state.embedding_model = load_embedding_model()
    app.state.collection = get_collection()
    app.state.gemini_client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )
    app.state.model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.1-flash-lite-preview",
    )
    app.state.graph = build_graph(
    collection=app.state.collection,
    embedding_model=app.state.embedding_model,
    client=app.state.gemini_client,
    model_name=app.state.model_name,
    )

    yield

    app.state.gemini_client.close()

app = FastAPI(
    title="DoubtDesk API",
    description="Multimodal PDF-based STEM tutor",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)