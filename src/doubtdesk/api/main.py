from doubtdesk.config import runtime_settings
import asyncio
import logging
from contextlib import suppress
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from doubtdesk.answering.output_guardrails import build_output_judge, close_output_judge_client
from doubtdesk.api.routes import router
from doubtdesk.conversations.routes import router as conversations_router
from doubtdesk.graph.builder import build_preparation_graph
from doubtdesk.api.streaming import PreparedAnswerGraph
from doubtdesk.db import initialize_database, close_pool
from doubtdesk.ingestion.embeddings import load_embedding_model
from doubtdesk.ingestion.ocr import ocr_available
from doubtdesk.retrieval.rerank import load_reranker
from doubtdesk.guardrails.scope import load_scope_classifier
from doubtdesk.guardrails.paraphrasing import BedrockChat, build_paraphraser
from doubtdesk.llm import preload_sdk_resources
from doubtdesk.analytics.routes import router as admin_router
from doubtdesk.auth.store import cleanup_expired_auth_records
from doubtdesk.quotas import cleanup_ask_requests
from doubtdesk.auth.routes import router as auth_router
from doubtdesk.auth.dependencies import allowed_origins
from doubtdesk.evals.routes import router as evals_router
from doubtdesk.evals.store import mark_interrupted
from doubtdesk.ingestion import jobs as ingestion_jobs
from doubtdesk.ingestion.registry import fail_stale_ingestions
from doubtdesk.observability.tracing import flush_traces
from doubtdesk.runtime import configure_threads


load_dotenv()
logger = logging.getLogger(__name__)


async def _periodic_auth_cleanup(interval=3600):
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(cleanup_expired_auth_records)
            await asyncio.to_thread(cleanup_ask_requests)
        except Exception:
            logger.exception("Periodic auth cleanup failed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = runtime_settings()
    configure_threads()
    initialize_database()
    mark_interrupted()
    fail_stale_ingestions()
    ingestion_jobs.remove_stale_uploads()
    cleanup_expired_auth_records()
    cleanup_ask_requests()
    app.state.scope_classifier = load_scope_classifier()
    app.state.embedding_model = load_embedding_model()
    app.state.reranker = load_reranker()
    app.state.ocr_available = await asyncio.to_thread(ocr_available)
    app.state.llm_client = AsyncOpenAI(api_key=config.openai_api_key, max_retries=2)
    app.state.model_name = config.openai_model
    app.state.paraphrase_client, app.state.paraphrase_model_name = build_paraphraser(
        config, app.state.llm_client, app.state.model_name,
    )
    app.state.assessment_model_name = config.evidence_assessment_model or app.state.model_name
    app.state.judge_client, app.state.output_judge = build_output_judge(config)
    paraphrase_sdk = app.state.paraphrase_client.client if isinstance(app.state.paraphrase_client, BedrockChat) else None
    await asyncio.to_thread(preload_sdk_resources, app.state.llm_client, paraphrase_sdk, app.state.judge_client)
    app.state.preparation_graph = build_preparation_graph(
        embedding_model=app.state.embedding_model,
        client=app.state.llm_client,
        model_name=app.state.model_name,
        scope_classifier=app.state.scope_classifier,
        paraphrase_model_name=app.state.paraphrase_model_name,
        assessment_model_name=app.state.assessment_model_name,
        reranker=app.state.reranker,
        paraphrase_client=app.state.paraphrase_client,
    )
    app.state.graph = PreparedAnswerGraph(app.state.preparation_graph, app.state.llm_client,
                                          app.state.model_name, app.state.output_judge)
    cleanup_task = asyncio.create_task(_periodic_auth_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        ingestion_jobs.shutdown()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await app.state.llm_client.close()
        if app.state.paraphrase_client is not app.state.llm_client:
            await app.state.paraphrase_client.close()
        await close_output_judge_client(app.state.judge_client)
        flush_traces()
        close_pool()

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
