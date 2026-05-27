import asyncio
import json
import http.client
import os

SERPER_KEY = os.environ.get('SERPER_API_KEY')

TOOL_SCHEMA = json.dumps({
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of query strings. Include multiple complementary search queries in a single call."
                }
            },
            "required": ["query"]
        }
    }
})


def google_search_with_serp(query: str, filter_huggingface: bool = False):
    def contains_chinese_basic(text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

    if contains_chinese_basic(query):
        payload = json.dumps({
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn"
        })
    else:
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en"
        })
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }

    results = None
    for i in range(5):
        try:
            conn = http.client.HTTPSConnection("google.serper.dev", timeout=20)
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            if not data:
                raise ValueError("empty response body from Serper")
            if res.status < 200 or res.status >= 300:
                raise ValueError(f"Serper HTTP {res.status}: {data.decode('utf-8', errors='ignore')[:200]}")
            results = json.loads(data.decode("utf-8"))
            break
        except Exception as e:
            print(f"[search_web] serper request/parse error attempt {i + 1}/5: {e}")
            if i == 4:
                return f"Google search Timeout, return None, Please try again later."
            continue

    if results is None:
        return f"Google search Timeout, return None, Please try again later."

    try:
        if "organic" not in results:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

        web_snippets = list()
        idx = 0
        if "organic" in results:
            for page in results["organic"]:
                link = str(page.get("link", ""))
                if filter_huggingface and ("huggingface" in link.lower()):
                    continue

                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = f"### A Google search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
        return content
    except:
        return f"No results found for '{query}'. Try with a more general query."



def search_with_serp(query: str):
    result = google_search_with_serp(query)
    return result

def _search_web_sync(query):
    if query is None:
        return "[Search] Invalid request format: missing 'query' field"
    if isinstance(query, str):
        response = search_with_serp(query)
    else:
        responses = [search_with_serp(q) for q in query]
        response = "\n\n---\n\n".join(responses)
    return response

async def search_web(query):
    return await asyncio.to_thread(_search_web_sync, query)
