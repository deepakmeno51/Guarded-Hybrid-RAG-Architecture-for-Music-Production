import os
from groq import Groq

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Missing GROQ_API_KEY environment variable. "
                "Required for the LlamaGuard safety check."
            )
        _client = Groq(api_key=api_key)
    return _client


class LlamaGuardChecker:
    def __init__(self, model: str = "llama-guard-3-8b"):
        self.model = model

    def check(self, text: str) -> tuple[bool, str | None]:
        client = _get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        verdict = resp.choices[0].message.content.strip()
        if verdict.startswith("unsafe"):
            category = verdict.split("\n")[-1] if "\n" in verdict else "unspecified"
            return True, category
        return False, None
