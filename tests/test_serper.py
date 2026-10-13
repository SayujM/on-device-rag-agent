import os
import sys
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def search_serper(query: str, num_results: int = 5):
    print("=" * 60)
    print("SERPER.DEV (GOOGLE FULL-WEB API) TEST")
    print("=" * 60)

    if not SERPER_API_KEY:
        print("[FAIL] 'SERPER_API_KEY' not found in environment or .env file.")
        print("Please add: SERPER_API_KEY=\"your_key_here\" to your .env file.")
        return None

    print(f"[*] API Key detected: {SERPER_API_KEY[:6]}...{SERPER_API_KEY[-4:]}")
    print(f"[*] Executing search query: '{query}'")

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results}).encode("utf-8")
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            body = response.read().decode("utf-8")
            data = json.loads(body)

            organic = data.get("organic", [])
            print(f"[PASS] Response status {status}. Received {len(organic)} organic Google results.\n")

            results = []
            for i, item in enumerate(organic, start=1):
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No Snippet")
                link = item.get("link", "")
                print(f"[{i}] {title}")
                print(f"    Snippet: {snippet}")
                print(f"    Link: {link}\n")
                results.append({"title": title, "snippet": snippet, "link": link})

            return results

    except urllib.error.HTTPError as e:
        print(f"[FAIL] HTTP Error: {e.code} - {e.reason}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Server response: {error_body}")
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return None


if __name__ == "__main__":
    test_query = "Toyota current global president and CEO 2026"
    res = search_serper(test_query)
    if res:
        print("=" * 60)
        print("Serper Google Full-Web API is 100% verified and operational!")
        print("=" * 60)
        sys.exit(0)
    else:
        sys.exit(1)
