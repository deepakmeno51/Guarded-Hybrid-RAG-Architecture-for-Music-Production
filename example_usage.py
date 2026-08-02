import asyncio
from guard_pipeline import InputGuard, FaithfulnessEvaluator, GuardedPipeline
from injection_guard import LlamaGuardChecker


async def llm_call(query: str, contexts: list[str]) -> str:
    prompt = f"""Context:
{chr(10).join(contexts)}

Question: {query}

Respond ONLY with JSON: {{"answer": str, "sources": [str], "confidence": float}}"""
    # replace with your actual generation call (Groq, OpenAI, etc.)
    return await your_llm_client.generate(prompt)


async def main():
    guard = InputGuard(checker=LlamaGuardChecker())
    evaluator = FaithfulnessEvaluator(threshold=0.7)
    pipeline = GuardedPipeline(guard, evaluator)

    query = "What's the refund policy?"
    retrieved_chunks = your_retriever.retrieve(query)  # your existing retrieval step

    result = await pipeline.process(query, llm_call, retrieved_chunks)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
