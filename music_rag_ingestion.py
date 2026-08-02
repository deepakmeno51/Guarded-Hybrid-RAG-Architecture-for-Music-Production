"""
music_rag_ingestion.py
Music Production Beginner RAG — Components 1-3:
  1. Data Ingestion Layer   (multi-source loaders)
  2. Preprocessing/Cleaning (boilerplate + transcript cleanup + dedup)
  3. Chunking Strategy      (source-aware splitting)

Install:
    pip install requests beautifulsoup4 youtube-transcript-api llama-index-core

Output:
    ./processed/chunks.jsonl  — ready for the embedding/vector-store stage (component 4+)
"""

import os
import re
import json
import hashlib
import logging
from enum import Enum
from typing import List, Optional, Dict

import requests
import certifi
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
except ImportError:
    YouTubeTranscriptApi = None  # handled gracefully at call site

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COMPONENT 1: DATA INGESTION LAYER
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    BLOG = "blog"            # official company blogs (iZotope, Sound on Sound, etc.)
    MANUAL = "manual"        # open-source DAW manuals (Ardour, LMMS)
    WIKIBOOK = "wikibook"    # CC BY-SA Wikibooks content
    TRANSCRIPT = "transcript"  # YouTube tutorial transcripts


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch raw HTML with basic error handling. Returns None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
            verify=certifi.where(),
        )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def load_blog_articles(urls: List[str], topic_map: Dict[str, str]) -> List[Document]:
    """Scrape official free blog articles (iZotope, Sound on Sound free tutorials, etc.)."""
    docs = []
    for url in urls:
        html = fetch_url(url)
        if not html:
            continue
        text = strip_boilerplate(html)
        if len(text) < 200:
            logger.warning(f"Skipping {url}: content too short after cleaning.")
            continue
        docs.append(Document(
            text=text,
            metadata={
                "source_url": url,
                "source_type": SourceType.BLOG.value,
                "topic": topic_map.get(url, "general"),
            },
        ))
    return docs


def load_manual_pages(urls: List[str], topic_map: Dict[str, str]) -> List[Document]:
    """Scrape open-source DAW manual pages (Ardour/LMMS docs — GPL/CC licensed)."""
    docs = []
    for url in urls:
        html = fetch_url(url)
        if not html:
            continue
        text = strip_boilerplate(html)
        if len(text) < 150:
            logger.warning(f"Skipping {url}: content too short after cleaning.")
            continue
        docs.append(Document(
            text=text,
            metadata={
                "source_url": url,
                "source_type": SourceType.MANUAL.value,
                "topic": topic_map.get(url, "general"),
            },
        ))
    return docs


def load_wikibooks(titles: List[str], topic_map: Dict[str, str], lang: str = "en") -> List[Document]:
    """Pull Wikibooks pages via the public MediaWiki API (CC BY-SA licensed)."""
    docs = []
    api_url = f"https://{lang}.wikibooks.org/w/api.php"
    for title in titles:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        try:
            resp = requests.get(
                api_url,
                params=params,
                timeout=15,
                headers={"User-Agent": "MusicRAGBot/1.0 (personal portfolio project)"},
                verify=certifi.where(),
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if len(extract) < 200:
                    logger.warning(f"Skipping Wikibook '{title}': little/no content.")
                    continue
                docs.append(Document(
                    text=clean_text(extract),
                    metadata={
                        "source_url": f"https://{lang}.wikibooks.org/wiki/{title}",
                        "source_type": SourceType.WIKIBOOK.value,
                        "topic": topic_map.get(title, "general"),
                    },
                ))
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning(f"Failed to fetch Wikibook '{title}': {e}")
    return docs


def _fetch_transcript_text(video_id: str) -> str:
    """Version-compatible transcript fetch.
    youtube-transcript-api <1.0 used a classmethod (get_transcript);
    >=1.0 switched to an instance method (fetch) returning snippet objects."""
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(seg["text"] for seg in raw)
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)
    return " ".join(snippet.text for snippet in fetched)


def load_youtube_transcripts(video_ids: List[str], topic_map: Dict[str, str]) -> List[Document]:
    """Pull public captions from YouTube videos. Personal/portfolio corpus use only —
    do not redistribute raw transcripts; ship the pipeline, not the scraped text."""
    if YouTubeTranscriptApi is None:
        logger.error("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
        return []
    docs = []
    for vid in video_ids:
        try:
            text = _fetch_transcript_text(vid)
            docs.append(Document(
                text=text,
                metadata={
                    "source_url": f"https://youtube.com/watch?v={vid}",
                    "source_type": SourceType.TRANSCRIPT.value,
                    "topic": topic_map.get(vid, "general"),
                },
            ))
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"No transcript available for video {vid}: {e}")
        except Exception as e:
            logger.warning(f"Failed to fetch transcript for {vid}: {e}")
    return docs


# ---------------------------------------------------------------------------
# COMPONENT 2: PREPROCESSING & CLEANING
# ---------------------------------------------------------------------------

def strip_boilerplate(html: str) -> str:
    """Remove nav/footer/ads/scripts; keep the main readable article text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup
    return clean_text(main.get_text(separator=" "))


def clean_text(text: str) -> str:
    """Generic whitespace/artifact cleanup shared by all source types."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[.*?\]", "", text)  # strip [Music], [Applause]-style transcript artifacts
    return text.strip()


FILLER_WORDS = {"um", "uh", "like", "you know", "kind of", "sort of"}


def clean_transcript(text: str) -> str:
    """Extra cleaning specific to spoken YouTube transcripts (filler-word removal)."""
    text = clean_text(text)
    words = text.split()
    filtered = [w for w in words if w.lower().strip(",.") not in FILLER_WORDS]
    return " ".join(filtered)


def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def deduplicate(documents: List[Document]) -> List[Document]:
    """Exact-content dedup via SHA256 hash of normalized text.

    NOTE: this catches identical/near-identical re-posts (common across mixing
    blogs that repeat the same tips). Embedding-based near-duplicate detection
    is a natural extension but is deferred to the embedding stage (component 4)
    to keep ingestion free of API dependencies.
    """
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        h = hashlib.sha256(_normalize_for_hash(doc.text).encode()).hexdigest()
        if h in seen_hashes:
            logger.info(f"Duplicate skipped: {doc.metadata.get('source_url')}")
            continue
        seen_hashes.add(h)
        unique_docs.append(doc)
    return unique_docs


# ---------------------------------------------------------------------------
# COMPONENT 3: CHUNKING STRATEGY (source-aware)
# ---------------------------------------------------------------------------

# Transcripts are conversational and repeat ideas -> smaller chunks.
# Manuals/blogs/wikibooks are denser/more structured -> larger chunks.
CHUNK_CONFIG = {
    SourceType.BLOG.value: dict(chunk_size=512, chunk_overlap=50),
    SourceType.MANUAL.value: dict(chunk_size=512, chunk_overlap=50),
    SourceType.WIKIBOOK.value: dict(chunk_size=512, chunk_overlap=50),
    SourceType.TRANSCRIPT.value: dict(chunk_size=256, chunk_overlap=30),
}


def chunk_documents(documents: List[Document]):
    """Apply source-aware SentenceSplitter chunking; returns LlamaIndex nodes."""
    all_nodes = []
    for doc in documents:
        source_type = doc.metadata.get("source_type", SourceType.BLOG.value)
        config = CHUNK_CONFIG.get(source_type, CHUNK_CONFIG[SourceType.BLOG.value])
        splitter = SentenceSplitter(**config)
        nodes = splitter.get_nodes_from_documents([doc])
        all_nodes.extend(nodes)
    logger.info(f"Chunked {len(documents)} documents into {len(all_nodes)} nodes.")
    return all_nodes


# ---------------------------------------------------------------------------
# ORCHESTRATION / EXAMPLE RUN
# ---------------------------------------------------------------------------

def main():
    # --- SOURCE LISTS: verified live URLs/titles as of this writing ---
    # (Sites restructure over time — re-check periodically.)

    # iZotope Learn blog — official, free, written by real audio engineers
    blog_urls = [
        "https://www.izotope.com/community/blog/what-is-mastering",
        "https://www.izotope.com/community/blog/how-to-eq-vocals",
        "https://www.izotope.com/community/blog/how-to-mix-music",
        "https://www.izotope.com/community/blog/mastering-for-streaming-platforms",
        "https://www.izotope.com/community/blog/digital-audio-basics-sample-rate-and-bit-depth",
        "https://www.izotope.com/community/blog/understanding-spectrograms",
        "https://www.izotope.com/community/blog/repairing-a-distorted-audio-track",
        "https://www.izotope.com/community/blog/removing-digital-clicks-and-pops-from-audio",
        "https://www.izotope.com/community/blog/vocal-doubler-and-tips-for-mixing-vocals",
        "https://www.izotope.com/community/blog/clean-up-vocals",
        "https://www.izotope.com/community/blog/fast-audio-cleanup",
        "https://www.izotope.com/community/blog/how-to-clean-up-audio-and-remove-background-noise",
        "https://www.izotope.com/community/blog/scene-rebalance",
    ]

    # Ardour Manual — open source DAW documentation (GPL/CC licensed)
    manual_urls = [
        "https://manual.ardour.org/mixing/",
        "https://manual.ardour.org/ardourmanual.html",
        "https://manual.ardour.org/welcome-to-ardour/",
        "https://manual.ardour.org/ardours-interface/about/",
        "https://manual.ardour.org/editing/edit-mode-and-tools/",
        "https://manual.ardour.org/editing/editing-basics/",
        "https://manual.ardour.org/editing-and-arranging/sections/",
        "https://manual.ardour.org/editing-and-arranging/create-region-fades-and-crossfades/",
        "https://manual.ardour.org/working-with-playlists/",
        "https://manual.ardour.org/working-with-playlists/understanding-playlists/",
        "https://manual.ardour.org/working-with-playlists/playlist-operations/",
        "https://manual.ardour.org/working-with-playlists/playlist_usecases/",
        "https://manual.ardour.org/automation/",
        "https://manual.ardour.org/recording/io-plugins/",
    ]

    # Wikibooks — CC BY-SA licensed, freely reusable via the MediaWiki API
    wikibook_titles = [
        "Mixing_and_Mastering",
        "Mixing_and_Mastering/Introduction",
        "Sound_Synthesis_Theory",
        "Sound_Synthesis_Theory/Introduction",
        "Sound_Synthesis_Theory/Sound_in_the_Digital_Domain",
        "Sound_Synthesis_Theory/Sound_in_the_Time_Domain",
        "Sound_Synthesis_Theory/Subtractive_Synthesis",
        "Sound_Synthesis_Theory/Modulation_Synthesis",
        "Sound_Synthesis_Theory/Physical_Modelling",
        "Sound_Synthesis_Theory/Synthesis_Software_and_Tools",
        "Sound_Synthesis_Theory/Links_and_Bibliography",
    ]

    # NOTE: load_youtube_transcripts() expects bare video IDs, not full URLs.
    youtube_video_ids: List[str] = [
        "1BLZGe-TqW0",
        "e0k0-o6R6eQ",
        "MwJYIzXTMHI",
        "yRzMby4PXzc",
    ]

    topic_map = {
        # blogs
        blog_urls[0]: "mastering",
        blog_urls[1]: "eq",
        blog_urls[2]: "mixing",
        blog_urls[3]: "mastering",
        blog_urls[4]: "digital_audio_basics",
        blog_urls[5]: "audio_repair",
        blog_urls[6]: "audio_repair",
        blog_urls[7]: "audio_repair",
        blog_urls[8]: "vocal_mixing",
        blog_urls[9]: "vocal_mixing",
        blog_urls[10]: "audio_repair",
        blog_urls[11]: "audio_repair",
        blog_urls[12]: "audio_repair",
        # manuals
        manual_urls[0]: "mixing",
        manual_urls[1]: "daw_overview",
        manual_urls[2]: "daw_overview",
        manual_urls[3]: "daw_interface",
        manual_urls[4]: "editing",
        manual_urls[5]: "editing",
        manual_urls[6]: "editing",
        manual_urls[7]: "editing",
        manual_urls[8]: "playlists",
        manual_urls[9]: "playlists",
        manual_urls[10]: "playlists",
        manual_urls[11]: "playlists",
        manual_urls[12]: "automation",
        manual_urls[13]: "recording",
        # wikibooks
        wikibook_titles[0]: "mixing",
        wikibook_titles[1]: "mixing",
        wikibook_titles[2]: "synthesis",
        wikibook_titles[3]: "synthesis",
        wikibook_titles[4]: "synthesis",
        wikibook_titles[5]: "synthesis",
        wikibook_titles[6]: "synthesis",
        wikibook_titles[7]: "synthesis",
        wikibook_titles[8]: "synthesis",
        wikibook_titles[9]: "synthesis",
        wikibook_titles[10]: "synthesis",
        # YouTube transcripts — placeholder topic; update once you know each
        # video's actual content (e.g. "mixing", "mastering", "vocal_production").
        youtube_video_ids[0]: "production",
        youtube_video_ids[1]: "production",
        youtube_video_ids[2]: "production",
        youtube_video_ids[3]: "production",
    }

    all_docs: List[Document] = []
    all_docs += load_blog_articles(blog_urls, topic_map)
    all_docs += load_manual_pages(manual_urls, topic_map)
    all_docs += load_wikibooks(wikibook_titles, topic_map)
    all_docs += load_youtube_transcripts(youtube_video_ids, topic_map)

    if not all_docs:
        logger.error("No documents ingested. Check network access and source URLs.")
        return

    logger.info(f"Ingested {len(all_docs)} raw documents.")

    # transcripts get extra filler-word cleaning after ingestion
    for doc in all_docs:
        if doc.metadata.get("source_type") == SourceType.TRANSCRIPT.value:
            doc.text = clean_transcript(doc.text)

    deduped_docs = deduplicate(all_docs)
    logger.info(f"{len(all_docs) - len(deduped_docs)} duplicate(s) removed.")

    nodes = chunk_documents(deduped_docs)

    # Persist chunks for the next stage (embedding + vector store, component 4+)
    os.makedirs("./processed", exist_ok=True)
    output_path = "./processed/chunks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps({
                "text": node.get_content(),
                "metadata": node.metadata,
            }) + "\n")

    logger.info(f"Saved {len(nodes)} chunks to {output_path}")


if __name__ == "__main__":
    main()
