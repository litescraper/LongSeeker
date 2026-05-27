import json
import os
import random
import time
import asyncio
import requests
import tiktoken
from tools.utils import call_server

VISIT_SERVER_TIMEOUT = 480
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv(
    "JINA_API_KEY"
)

_p = os.environ.get("VISIT_HTTP_PROXY").strip()
VISIT_REQUESTS_PROXIES = {"http": _p, "https": _p} if _p else None

TOOL_SCHEMA = json.dumps({
    "type": "function",
    "function": {
        "name": "visit_web",
        "description": "Visit webpage(s) and return the summary of the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                },
                "goal": {
                    "type": "string",
                    "description": "The goal of the visit for webpage(s)."
                }
            },
            "required": ["url", "goal"]
        }
    }
})

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def jina_readpage(url: str) -> str:
    """
    Read webpage content using Jina service.

    Args:
        url: The URL to read
        goal: The goal/purpose of reading the page

    Returns:
        str: The webpage content or error message
    """
    max_retries = 3
    timeout = 50

    for attempt in range(max_retries):
        headers = {
            "Authorization": f"Bearer {JINA_API_KEYS}",
        }
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=timeout,
                proxies=VISIT_REQUESTS_PROXIES,
            )
            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            else:
                raise ValueError("jina readpage error")
        except Exception as e:
            time.sleep(0.5)
            if attempt == max_retries - 1:
                return "[visit] Failed to read page."

    return "[visit] Failed to read page."


def html_readpage_jina(url: str) -> str:
    max_attempts = 8
    for attempt in range(max_attempts):
        content = jina_readpage(url)
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            return content
    return "[visit] Failed to read page."


def readpage_jina(url: str, goal: str) -> str:
    """
    Attempt to read webpage content by alternating between jina and aidata services.

    Args:
        url: The URL to read
        goal: The goal/purpose of reading the page

    Returns:
        str: The webpage content or error message
    """
    max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 3))

    content = html_readpage_jina(url)

    if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
        content = truncate_to_tokens(content, max_tokens=95000)
        messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
        parse_retry_times = 0
        raw = call_server(messages, max_retries=max_retries)
        summary_retries = 2
        while len(raw) < 10 and summary_retries > 0:
            truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
            status_msg = (
                f"[visit] Summary url[{url}] "
                f"attempt {2 - summary_retries + 1}/2, "
                f"content length: {len(content)}, "
                f"truncating to {truncate_length} chars"
            ) if summary_retries > 0 else (
                f"[visit] Summary url[{url}] failed after 2 attempts, "
                f"final truncation to 25000 chars"
            )
            print(status_msg)
            content = content[:truncate_length]
            extraction_prompt = EXTRACTOR_PROMPT.format(
                webpage_content=content,
                goal=goal
            )
            messages = [{"role": "user", "content": extraction_prompt}]
            raw = call_server(messages, max_retries=max_retries)
            summary_retries -= 1

        parse_retry_times = 2
        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()
        while parse_retry_times < 3:
            try:
                raw = json.loads(raw)
                break
            except:
                raw = call_server(messages, max_retries=max_retries)
                parse_retry_times += 1

        if parse_retry_times >= 3:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                url=url, goal=goal
            )
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                url=url, goal=goal
            )
            useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
            useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

        if len(useful_information) < 10 and summary_retries < 0:
            print("[visit] Could not generate valid summary after maximum retries")
            useful_information = "[visit] Failed to read page"

        return useful_information
    else:
        return f"[visit] Failed to visit the url: {url}"


async def visit_web(url, goal):
    if url is None or goal is None:
        return "[Visit] Invalid request format: missing 'url' or 'goal' field"

    start_time = time.time()

    # Create log folder if it doesn't exist
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)

    if isinstance(url, str):
        response = await asyncio.to_thread(readpage_jina, url, goal)
    else:
        response = []
        start_time = time.time()
        for u in url:
            if time.time() - start_time > 900:
                cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                try:
                    cur_response = await asyncio.to_thread(readpage_jina, u, goal)
                except Exception as e:
                    cur_response = f"Error fetching {u}: {str(e)}"
            response.append(cur_response)
        response = "\n---\n".join(response)

    return response.strip()
