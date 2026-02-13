import os
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv

def search_bing_serpapi(query: str, limit: int = 10):
    """
    Returns:
      - list of SERP result links
      - google-style link like: https://www.google.com/search?q=abid+ali+awan
    """
    load_dotenv()
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_KEY is not set in .env")

    # Build google.com/search?q=...
    google_style_link = f"https://www.google.com/search?q={quote_plus(query)}"

    # Call SerpAPI (Bing engine)
    resp = requests.get(
        "https://serpapi.com/search",
        params={"engine": "bing", "q": query, "api_key": api_key, "num": limit},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    # Extract SERP links
    links = []
    for item in payload.get("organic_results", []):
        link = item.get("link")
        if link:
            links.append(link)

    return links, google_style_link


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    query = "pm modi"

    serp_links, google_link = search_bing_serpapi(query)

    print("🔗 Google‑style search link:")
    print(google_link)

    print("\n🔗 SERP results (from Bing via SerpAPI):")
    for l in serp_links:
        print("-", l)