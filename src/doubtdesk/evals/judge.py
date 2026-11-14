from doubtdesk.config import runtime_settings
import math
from functools import lru_cache
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness,
)
from doubtdesk.rate_limit import RequestPacer


@lru_cache(maxsize=1)
def get_judge_pacer():
    return RequestPacer(runtime_settings().judge_rpm)


class RagasJudge:
    def __init__(self):
        api_key = runtime_settings().gemini_api_key
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.llm = llm_factory(
            runtime_settings().ragas_judge_model,
            provider="openai",
            client=self.client,
            adapter="instructor",
        )
        embeddings = embedding_factory(
            "huggingface",
            model=runtime_settings().judge_embed_model,
            interface="modern",
        )
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
