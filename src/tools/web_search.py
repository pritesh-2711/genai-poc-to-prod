"""
Web search and webpage fetching tools.

Dependencies:
    pip install tavily-python httpx beautifulsoup4

Environment variable required:
    TAVILY_API_KEY
"""

import os
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

from ..core.logging import LoggingManager

logger = LoggingManager.get_logger(__name__)

_TAVILY_BASE_URL = "https://api.tavily.com"
_DEFAULT_SEARCH_DEPTH = "advanced"
_DEFAULT_MAX_RESULTS = 5
_FETCH_TIMEOUT_SECONDS = 15
_MAX_CONTENT_CHARS = 8000

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _get_tavily_api_key() -> str:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is not set. "
            "Get a key at https://tavily.com and add it to your .env file."
        )
    return key


@tool
def web_search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    include_domains: Optional[list[str]] = None,
    search_depth: str = _DEFAULT_SEARCH_DEPTH,
) -> str:
    """Search the web for current information about research topics, papers, authors, or concepts.

    Use this when:
    - The user asks about recent papers or findings not in uploaded documents.
    - You need to find the source, citation count, or publication details of a paper.
    - The user asks about a concept or author you need external context on.
    - You want to verify or supplement information from uploaded documents.

    Do NOT use this when the answer is clearly contained in the uploaded documents.

    Args:
        query: A focused search query. Be specific — include paper titles, author names,
               or technical terms rather than vague phrases.
        max_results: Number of results to return. Default 5, max 10.
        include_domains: Optional list of domains to restrict results to.
                         Example: ["arxiv.org", "semanticscholar.org", "papers.nips.cc"]
        search_depth: "basic" for fast results or "advanced" for deeper research. Default "advanced".

    Returns:
        Formatted string of search results with titles, URLs, and content snippets.
    """
    api_key = _get_tavily_api_key()
    max_results = min(max_results, 10)

    payload: dict = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }

    if include_domains:
        payload["include_domains"] = include_domains

    logger.info("web_search: query=%r depth=%s max_results=%d", query, search_depth, max_results)

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{_TAVILY_BASE_URL}/search", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        logger.error("Tavily API HTTP error: %s", e)
        return f"Search failed: HTTP {e.response.status_code} from Tavily API."
    except httpx.RequestError as e:
        logger.error("Tavily API request error: %s", e)
        return f"Search failed: Could not reach Tavily API. {e}"

    results = data.get("results", [])
    if not results:
        return f"No results found for query: {query!r}"

    lines: list[str] = []

    tavily_answer = data.get("answer", "")
    if tavily_answer:
        lines.append(f"Summary: {tavily_answer}\n")

    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")
        score = result.get("score", 0.0)

        lines.append(f"[{i}] {title}")
        lines.append(f"    URL: {url}")
        lines.append(f"    Relevance: {score:.2f}")
        if content:
            snippet = content[:300].replace("\n", " ")
            lines.append(f"    Excerpt: {snippet}...")
        lines.append("")

    return "\n".join(lines)


@tool
def fetch_webpage(url: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """Fetch and extract the text content of a webpage or online paper.

    Use this after web_search when you need the full content of a specific result,
    such as reading an abstract page on arXiv, a paper landing page, or a blog post
    about a research topic.

    Works well with:
    - arXiv abstract pages (e.g. https://arxiv.org/abs/1706.03762)
    - Semantic Scholar paper pages
    - Research blog posts and documentation pages

    Does NOT work with:
    - PDF files (use uploaded documents instead)
    - Pages behind login walls or paywalls
    - JavaScript-rendered pages

    Args:
        url: The full URL to fetch. Must start with http:// or https://.
        max_chars: Maximum characters to return from the page content. Default 8000.

    Returns:
        Extracted plain text from the page, truncated to max_chars.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url!r}. URL must start with http:// or https://"

    if url.lower().endswith(".pdf"):
        return (
            "This URL points to a PDF file. Please download and upload the PDF directly "
            "to the session so it can be properly extracted and indexed."
        )

    logger.info("fetch_webpage: url=%r", url)

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ResearchAssistantBot/1.0; "
                "+https://github.com/pritesh-2711/genai-poc-to-prod)"
            )
        }
        with httpx.Client(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPStatusError as e:
        return f"Failed to fetch page: HTTP {e.response.status_code} at {url}"
    except httpx.RequestError as e:
        return f"Failed to fetch page: {e}"

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)
    except ImportError:
        import re
        content = re.sub(r"<[^>]+>", " ", html)
        content = re.sub(r"\s+", " ", content).strip()

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[Truncated at {max_chars} characters]"

    if not content.strip():
        return f"No readable text content found at {url}"

    return f"Content from {url}:\n\n{content}"
