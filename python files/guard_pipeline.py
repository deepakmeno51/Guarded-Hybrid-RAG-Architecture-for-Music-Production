import json
import asyncio
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from pydantic import BaseModel, field_validator
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from injection_guard import LlamaGuardChecker
from output_schema import RAGResponse

_analyzer = AnalyzerEngine()


class GuardResult(BaseModel):
    blocked: bool
    reason: str | None = None
    sanitized_text: str

    @field_validator("sanitized_text")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("empty sanitized_text")
        return v


@dataclass
class InputGuard:
    checker: LlamaGuardChecker

    def redact_pii(self, text: str) -> str:
        results = _analyzer.analyze(text=text, language="en")
        redacted = text
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            redacted = redacted[: r.start] + f"[{r.entity_type}]" + redacted[r.end :]
        return redacted

    def run(self, text: str) -> GuardResult:
        is_unsafe, category = self.checker.check(text)
        if is_unsafe:
            return GuardResult(blocked=True, reason=f"unsafe:{category}", sanitized_text=text)
        clean = self.redact_pii(text)
        return GuardResult(blocked=False, sanitized_text=clean)


class FaithfulnessEvaluator:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    async def score(self, question: str, answer: str, contexts: list[str]) -> float:
        loop = asyncio.get_event_loop()
        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })
        result = await loop.run_in_executor(None, lambda: evaluate(ds, metrics=[faithfulness]))
        return float(result["faithfulness"][0])


class GuardedPipeline:
    def __init__(self, guard: InputGuard, evaluator: FaithfulnessEvaluator):
        self.guard = guard
        self.evaluator = evaluator

    async def process(self, user_input: str, llm_call, contexts: list[str]) -> dict:
        guard_result = self.guard.run(user_input)
        if guard_result.blocked:
            return {"status": "blocked", "reason": guard_result.reason}

        raw_output = await llm_call(guard_result.sanitized_text, contexts)
        try:
            structured = RAGResponse.model_validate(json.loads(raw_output))
        except Exception as e:
            return {"status": "schema_invalid", "error": str(e)}

        score = await self.evaluator.score(guard_result.sanitized_text, structured.answer, contexts)

        return {
            "status": "passed" if score >= self.evaluator.threshold else "flagged",
            "response": structured.model_dump(),
            "faithfulness": score,
        }
