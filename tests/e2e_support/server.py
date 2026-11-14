import chromadb
import os
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from doubtdesk.analytics.store import initialize_analytics_db
from doubtdesk.api.routes import router
from doubtdesk.auth.routes import router as auth_router
from doubtdesk.auth.schemas import SignupRequest
from doubtdesk.auth.service import register_user
from doubtdesk.auth.store import initialize_auth_db
from doubtdesk.graph.builder import build_preparation_graph
from doubtdesk.api.streaming import PreparedAnswerGraph
from doubtdesk.ingestion.embeddings import load_embedding_model
from doubtdesk.ingestion.registry import initialize_registry
from doubtdesk.memory.store import initialize_memory_db
from doubtdesk.retrieval.vector_store import get_collection
from tests.e2e_support.fake_openai import FakeOpenAIClient, FakeOutputJudge
from doubtdesk.conversations.routes import router as conversations_router
from doubtdesk.memory.vector import get_memory_collection


test_directory = tempfile.TemporaryDirectory(
    prefix="doubtdesk_e2e_",
    ignore_cleanup_errors=True,
)
os.chdir(test_directory.name)
Path("data").mkdir(exist_ok=True)


class FakeScopeClassifier:
    def predict(self, questions: list[str]) -> list[dict]:
        return [
            {
                "question": question,
                "label": 0,
                "off_topic_probability": 0.0,
            }
            for question in questions
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_registry()
    initialize_memory_db()
    initialize_analytics_db()
    initialize_auth_db()

    test_user = register_user(
        SignupRequest(
            username="e2e_student",
            email="e2e@example.com",
            password="e2e-password-123",
        )
    )
    with closing(sqlite3.connect("data/auth.db")) as conn, conn:
        conn.execute(
            "UPDATE users SET verified_at = ? WHERE user_id = ?",
            (int(time.time()), test_user["user_id"]),
        )

    app.state.embedding_model = load_embedding_model()
    app.state.chroma_client = chromadb.PersistentClient(path="data/chroma")
    app.state.collection = get_collection(client=app.state.chroma_client)
    app.state.llm_client = FakeOpenAIClient()
    app.state.output_judge = FakeOutputJudge()
    app.state.model_name = "fake-openai"
    app.state.memory_collection = get_memory_collection(client=app.state.chroma_client)
    app.state.scope_classifier = FakeScopeClassifier()
    app.state.preparation_graph = build_preparation_graph(
        collection=app.state.collection,
        embedding_model=app.state.embedding_model,
        client=app.state.llm_client,
        model_name=app.state.model_name,
        scope_classifier=app.state.scope_classifier,
        memory_collection=app.state.memory_collection,
    )
    app.state.graph = PreparedAnswerGraph(app.state.preparation_graph, app.state.llm_client,
                                          app.state.model_name, app.state.output_judge)

    try:
        yield
    finally:
        app.state.llm_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)
app.include_router(router)
app.include_router(auth_router)

app.include_router(conversations_router)
