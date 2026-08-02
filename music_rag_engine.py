"""
music_rag_engine.py
Query engine for the Music Production Beginner RAG project.

Wires together:
  - Components 4-6 (this project): embeddings + ChromaDB + retriever, reading
    the ./processed/chunks.jsonl produced by music_rag_ingestion.py
  - Guarded pipeline (uploaded): input safety check (LlamaGuard) + PII
    redaction (Presidio) -> generation -> structured output validation
    (Pydantic) -> faithfulness scoring (RAGAS)

Install:
    pip install llama-index-core llama-index-embeddings-openai llama-index-vector-stores-chroma chromadb
    pip install groq ragas==0.2.10 datasets "pydantic>=2.0" presidio-analyzer presidio-anonymizer python-dotenv
    python -m spacy download en_core_web_lg   # required by presidio-analyzer at import time

Environment variables required (.env or exported):
    OPENAI_API_KEY   - used for embeddings only (swap for a local embedder to drop this dependency)
    GROQ_API_KEY     - used for BOTH generation (llama-3.1-8b-instant) and the LlamaGuard safety check

Prerequisite:
    Run music_rag_ingestion.py first so ./processed/chunks.jsonl exists.
"""

import os
import sys
import json
import asyncio
import logging

from dotenv import load_dotenv

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from groq import Groq

from guard_pipeline import InputGuard, FaithfulnessEvaluator, GuardedPipeline
from injection_guard import LlamaGuardChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = "./processed/chunks.jsonl"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "music_rag_collection"
TOP_K = 3
GENERATION_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# ENV / FILE VALIDATION
# ---------------------------------------------------------------------------
def load_and_validate_env() -> None:
    load_dotenv()
    missing = [k for k in ("OPENAI_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]
    if missing:
        logger.error(f"Missing required environment variable(s): {', '.join(missing)}")
        sys.exit(1)


def load_chunks(path: str) -> list[Document]:
    if not os.path.isfile(path):
        logger.error(f"'{path}' not found. Run music_rag_ingestion.py first.")
        sys.exit(1)
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(text=row["text"], metadata=row.get("metadata", {})))
    if not docs:
        logger.error(f"'{path}' is empty. Nothing to index.")
        sys.exit(1)
    logger.info(f"Loaded {len(docs)} chunks from {path}.")
    return docs


# ---------------------------------------------------------------------------
# INDEX / RETRIEVER (components 4-6)
# ---------------------------------------------------------------------------
def build_retriever() -> VectorIndexRetriever:
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() > 0:
        logger.info("Existing embeddings found — loading index from ChromaDB.")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )
    else:
        docs = load_chunks(CHUNKS_PATH)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        logger.info("Index built and persisted to disk.")

    return VectorIndexRetriever(index=index, similarity_top_k=TOP_K)


# ---------------------------------------------------------------------------
# GENERATION (Groq, forced structured JSON matching RAGResponse schema)
# ---------------------------------------------------------------------------
_groq_client = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


QA_SYSTEM_PROMPT = """You are a beginner-friendly music production assistant. Answer using ONLY the Context below.

Rules:
- Rely ONLY on facts clearly stated in the Context.
- Do NOT assume, extrapolate, or use outside knowledge.
- If the Context does not contain the answer, set "answer" to "Data not found." and "confidence" to 0.0.
- Avoid jargon unless the Context itself defines the term first.
- Each context block below starts with its source in [brackets]. In "sources",
  include ONLY the exact bracketed source strings you actually used.

Respond ONLY with valid JSON, no other text, matching exactly:
{"answer": str, "sources": [str, ...], "confidence": float between 0.0 and 1.0}"""


async def llm_call(query: str, contexts: list[str]) -> str:
    """Generation step, invoked by GuardedPipeline only after input safety checks pass."""
    context_block = "\n\n---\n\n".join(contexts)
    loop = asyncio.get_event_loop()

    def _call():
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ],
        )
        return resp.choices[0].message.content

    return await loop.run_in_executor(None, _call)


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------
async def _run_query(pipeline: GuardedPipeline, retriever: VectorIndexRetriever, query: str) -> None:
    try:
        nodes = retriever.retrieve(query)
        if not nodes:
            print("[Answer] Data not found.\n")
            return
        # tag each chunk with its source so the LLM can cite it and we can
        # verify citations aren't invented
        contexts = [
            f"[{n.metadata.get('source_url', n.metadata.get('source_type', 'unknown'))}]\n{n.get_content()}"
            for n in nodes
        ]
        result = await pipeline.process(query, llm_call, contexts)
        print(f"[Result] {json.dumps(result, indent=2)}\n")
    except Exception as e:
        logger.error(f"Query failed: {e}")


async def main():
    load_and_validate_env()

    try:
        retriever = build_retriever()
    except Exception as e:
        logger.error(f"Fatal error building retriever: {e}")
        sys.exit(1)

    try:
        guard = InputGuard(checker=LlamaGuardChecker())
        evaluator = FaithfulnessEvaluator(threshold=0.7)
        pipeline = GuardedPipeline(guard, evaluator)
    except Exception as e:
        logger.error(f"Fatal error initializing guard pipeline: {e}")
        logger.error("Check that GROQ_API_KEY is set and 'python -m spacy download en_core_web_lg' has been run.")
        sys.exit(1)

    print("\n=== Music Production Archivist (Guarded) Ready ===")
    print("Type a question, or 'exit' to quit.\n")

    example_question = "What's the difference between mixing and mastering?"
    print(f"[Example Query] {example_question}")
    await _run_query(pipeline, retriever, example_question)

    while True:
        try:
            user_input = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not user_input:
            continue
        await _run_query(pipeline, retriever, user_input)


if __name__ == "__main__":
    asyncio.run(main())
