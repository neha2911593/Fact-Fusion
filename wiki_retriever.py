import requests
import urllib.parse

def get_evidence(claim):

    # -----------------------------  
    # 1. BETTER WIKIPEDIA SEARCH  
    # -----------------------------  
    try:
        # Convert claim → short keyword query
        # Example: "CO2 human activity climate change"
        keywords = " ".join(claim.split()[:6])

        encoded = urllib.parse.quote(keywords)

        search_url = f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={encoded}&limit=1"
        res = requests.get(search_url, timeout=5).json()

        if "pages" in res and res["pages"]:
            page_id = res["pages"][0]["id"]

            summary_url = f"https://en.wikipedia.org/w/rest.php/v1/page/{page_id}/summary"
            summary = requests.get(summary_url, timeout=5).json()

            if "extract" in summary:
                return f"Wikipedia Evidence: {summary['extract']}"

    except Exception as e:
        print("Wikipedia error:", e)


    # -------------------------------------------------------------------
    # 2. NASA CLIMATE DATA API (Fallback 1)
    # -------------------------------------------------------------------
    try:
        nasa_url = "https://climate.nasa.gov/api/v1/news"
        nasa_res = requests.get(nasa_url, timeout=5).json()

        if "items" in nasa_res and nasa_res["items"]:
            item = nasa_res["items"][0]
            title = item.get("title", "")
            desc = item.get("description", "")

            return f"NASA Climate Evidence: {title} — {desc}"
    except Exception as e:
        print("NASA API error:", e)

    # -------------------------------------------------------------------
    # 3. DUCKDUCKGO INSTANT ANSWER API (Free) (Fallback 2)
    # -------------------------------------------------------------------
    try:
        ddg_url = f"https://api.duckduckgo.com/?q={claim}&format=json&no_redirect=1"
        ddg_res = requests.get(ddg_url, timeout=5).json()

        abstract = ddg_res.get("AbstractText", "")
        if abstract:
            return f"DuckDuckGo Evidence: {abstract}"
    except Exception as e:
        print("DuckDuckGo API error:", e)

    # -------------------------------------------------------------------
    # 4. OPEN-METEO CLIMATE INDICATORS API (Fallback 3)
    #     Good for climate-related claims
    # -------------------------------------------------------------------
    try:
        climate_url = "https://climate-api.open-meteo.com/v1/climate?latitude=0&longitude=0&start_year=1990&end_year=2020"
        climate_res = requests.get(climate_url, timeout=5).json()

        if "temperature_2m" in climate_res:
            return "Climate Data Evidence: Global climate trend data fetched successfully."
    except Exception as e:
        print("Open-Meteo API error:", e)

    # -------------------------------------------------------------------
    # 5. NEWS API (Optional – requires API key) (Fallback 4)
    # -------------------------------------------------------------------
    # try:
    #     NEWS_KEY = "YOUR_API_KEY"
    #     news_url = f"https://newsapi.org/v2/everything?q={claim}&apiKey={NEWS_KEY}"
    #     news_res = requests.get(news_url, timeout=5).json()

    #     if "articles" in news_res and news_res["articles"]:
    #         article = news_res["articles"][0]
    #         return f"News Evidence: {article['title']} — {article['description']}"
    # except:
    #     pass

    # -------------------------------------------------------------------
    # FINAL RESORT
    # -------------------------------------------------------------------
    return "No relevant evidence found."
