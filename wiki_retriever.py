import wikipedia
import requests
from wikipedia.exceptions import DisambiguationError, PageError

def get_evidence(claim):
    # ---------------- WIKIPEDIA ----------------
    try:
        results = wikipedia.search(claim)
        if results:
            try:
                summary = wikipedia.summary(results[0], sentences=3)
                return f"Wikipedia: {summary}"
            except DisambiguationError as e:
                # Choose first disambiguation option
                summary = wikipedia.summary(e.options[0], sentences=3)
                return f"Wikipedia (disambiguation): {summary}"
            except PageError:
                return "Wikipedia page not found for this claim."
    except Exception as e:
        print("Wikipedia error:", e)

    # ---------------- NASA FALLBACK ----------------
    try:
        url = "https://climate.nasa.gov/api/v1/news"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            return f"NASA Climate Data: {item['title']} - {item['description']}"
    except Exception as e:
        print("NASA API error:", e)

    return "No relevant evidence found."
