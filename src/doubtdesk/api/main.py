import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from google import genai
from doubtdesk.answering.output_guardrails import GeminiOutputJudge
from doubtdesk.api.routes import router
from doubtdesk.conversations.routes import router as conversations_router
from doubtdesk.graph.builder import build_graph, build_preparation_graph
from doubtdesk.ingestion.embeddings import load_embedding_model
from doubtdesk.ingestion.registry import initialize_registry
from doubtdesk.retrieval.vector_store import get_collection
from doubtdesk.retrieval.lexical_store import backfill_missing_documents
from doubtdesk.guardrails.scope import load_scope_classifier
from doubtdesk.memory.store import initialize_memory_db
from doubtdesk.memory.vector import get_memory_collection
from doubtdesk.analytics.store import initialize_analytics_db
from doubtdesk.analytics.routes import router as admin_router
from doubtdesk.auth.store import initialize_auth_db, cleanup_expired_auth_records
from doubtdesk.auth.routes import router as auth_router
from doubtdesk.auth.dependencies import allowed_origins
from doubtdesk.evals.routes import router as evals_router
from doubtdesk.evals.store import initialize_evals_db, mark_interrupted
from doubtdesk.observability.tracing import flush_traces


load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_registry()
    initialize_memory_db()
    initialize_analytics_db()
    initialize_auth_db()
    initialize_evals_db()
    mark_interrupted()
    cleanup_expired_auth_records()
    app.state.scope_classifier = load_scope_classifier()
    app.state.embedding_model = load_embedding_model()
    app.state.collection = get_collection()
    backfill_missing_documents(app.state.collection)
    app.state.memory_collection = get_memory_collection()
    app.state.llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=2)
    app.state.model_name = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    app.state.paraphrase_model_name = os.getenv("PARAPHRASE_MODEL") or app.state.model_name
    app.state.assessment_model_name = os.getenv("EVIDENCE_ASSESSMENT_MODEL") or app.state.model_name
    app.state.gemini_judge_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    app.state.output_judge = GeminiOutputJudge(app.state.gemini_judge_client)
    app.state.graph = build_graph(
        collection=app.state.collection,
        embedding_model=app.state.embedding_model,
        client=app.state.llm_client,
        model_name=app.state.model_name,
        scope_classifier=app.state.scope_classifier,
        memory_collection=app.state.memory_collection,
        paraphrase_model_name=app.state.paraphrase_model_name,
        assessment_model_name=app.state.assessment_model_name,
        output_judge=app.state.output_judge,
    )
    app.state.preparation_graph = build_preparation_graph(
        collection=app.state.collection,
        embedding_model=app.state.embedding_model,
        client=app.state.llm_client,
        model_name=app.state.model_name,
        scope_classifier=app.state.scope_classifier,
        memory_collection=app.state.memory_collection,
        paraphrase_model_name=app.state.paraphrase_model_name,
        assessment_model_name=app.state.assessment_model_name,
    )
    yield
    app.state.llm_client.close()
    app.state.gemini_judge_client.close()
    flush_traces()

app = FastAPI(
    title="DoubtDesk API",
    description="Multimodal PDF-supported academic tutor",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins()),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-CSRF-Token", "X-Admin-Key"],
)
app.include_router(router)
app.include_router(conversations_router)
app.include_router(admin_router)
app.include_router(auth_router)
app.include_router(evals_router)
