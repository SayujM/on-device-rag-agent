import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


def search_duckduckgo(query: str, max_results: int = 5) -> Optional[List[Dict[str, str]]]:
    """
    Tier 1: DuckDuckGo free search (zero quota, unauthenticated).
    Returns None if rate-limited, throttled, or unavailable.
    """
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        raw_results = ddgs.text(query, max_results=max_results)
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", "") or r.get("snippet", ""),
                "link": r.get("href", "") or r.get("link", ""),
            })
        if results:
            return results
        return None
    except Exception as e:
        print(f"[Tier 1 DuckDuckGo Throttled/Error]: {e}")
        return None


def search_serper(query: str, max_results: int = 5) -> Optional[List[Dict[str, str]]]:
    """
    Tier 2: Serper.dev Google Full-Web API.
    Used when DuckDuckGo is rate-limited or fails.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("[Tier 2 Serper]: SERPER_API_KEY not configured in .env.")
        return None

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": max_results}).encode("utf-8")
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                organic = data.get("organic", [])
                results = []
                for item in organic:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", "")
                    })
                return results if results else None
    except Exception as e:
        print(f"[Tier 2 Serper Error]: {e}")
        return None


def hybrid_web_search(query: str, max_results: int = 5) -> Tuple[List[Dict[str, str]], str]:
    """
    Tiered Hybrid Web Search:
    1. Attempts DuckDuckGo first (free, unconstrained whole-web search).
    2. If DuckDuckGo is rate-limited, throttled, or fails, automatically fails over to Serper (Google Full-Web API).
    Returns a tuple of (results, engine_used).
    """
    print(f"\n[Hybrid Web Search] Query: '{query}'")

    # Try Tier 1: DuckDuckGo
    print("  -> Attempting Tier 1: DuckDuckGo...")
    ddg_results = search_duckduckgo(query, max_results=max_results)
    if ddg_results:
        print(f"  [PASS] DuckDuckGo returned {len(ddg_results)} results.")
        return ddg_results, "DuckDuckGo"

    # Failover to Tier 2: Serper (Google)
    print("  [WARN] DuckDuckGo throttled or returned 0 results. Failing over to Tier 2: Google Search (Serper)...")
    serper_results = search_serper(query, max_results=max_results)
    if serper_results:
        print(f"  [PASS] Google Search (Serper) returned {len(serper_results)} results.")
        return serper_results, "Google Search (via Serper)"

    print("  [FAIL] Both search tiers returned no results.")
    return [], "None"
