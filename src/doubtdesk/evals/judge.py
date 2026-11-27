from doubtdesk.config import runtime_settings
import math
from functools import lru_cache
from doubtdesk.llm import bedrock_chat_client
from ragas.llms import llm_factory
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness,
)
from doubtdesk.ingestion.embeddings import EMBEDDING_MODEL_NAME, OnnxEmbeddingModel
from doubtdesk.rate_limit import RequestPacer


@lru_cache(maxsize=1)
def get_judge_pacer():
    return RequestPacer(runtime_settings().judge_rpm)


class SharedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name: str, model_instance):
        self._shared_instance = model_instance
        super().__init__(model=model_name)

    def _setup_local_model(self):
        self.model_instance = self._shared_instance


def judge_embeddings(model_name: str, embedding_model=None):
    if embedding_model is not None and model_name.removeprefix("sentence-transformers/") == EMBEDDING_MODEL_NAME:
        return SharedHuggingFaceEmbeddings(model_name, embedding_model)
    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return SharedHuggingFaceEmbeddings(model_name, OnnxEmbeddingModel(repo))


class RagasJudge:
    def __init__(self, embedding_model=None):
        self.client = bedrock_chat_client(runtime_settings(), "running RAGAS evaluations")
        self.llm = llm_factory(
            runtime_settings().ragas_judge_model,
            provider="openai",
            client=self.client,
            adapter="instructor",
        )
        embeddings = judge_embeddings(runtime_settings().judge_embed_model, embedding_model)
        self.metrics = {
            "context_precision": ContextPrecision(llm=self.llm),
            "context_recall": ContextRecall(llm=self.llm),
            "faithfulness": Faithfulness(llm=self.llm),
            "answer_relevancy": AnswerRelevancy(llm=self.llm, embeddings=embeddings),
            "answer_correctness": AnswerCorrectness(llm=self.llm, embeddings=embeddings),
        }

    async def score(self, *, question, contexts, reference, requirements, answer=None, thresholds):
        values = {}
        errors = {}
        params = {
            "context_precision": {
                "user_input": question, "retrieved_contexts": contexts, "reference": reference,
            },
            "context_recall": {
                "user_input": question, "retrieved_contexts": contexts,
                "reference": " ".join(requirements),
            },
        }
        if answer is not None:
            params.update({
                "faithfulness": {"user_input": question, "response": answer, "retrieved_contexts": contexts},
                "answer_relevancy": {"user_input": question, "response": answer},
                "answer_correctness": {"user_input": question, "response": answer, "reference": reference},
            })
        for name, arguments in params.items():
            try:
                await get_judge_pacer().acquire_async()
                result = await self.metrics[name].ascore(**arguments)
                value = float(result.value)
                if not math.isfinite(value):
                    raise ValueError("Judge score is not finite")
                values[name] = value
            except Exception as exc:
                values[name] = None
                errors[name] = f"{type(exc).__name__}: {str(exc)[:240]}"
        passed = {name: value is not None and value >= getattr(thresholds, name) for name, value in values.items()}
        return {"scores": values, "passed": passed, "all_passed": all(passed.values()), "metric_errors": errors}

    async def close(self):
        await self.client.close()
