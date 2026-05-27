import json
import os
from openai import OpenAI

def call_server(msgs, max_retries=2):
    client = OpenAI(
        api_key=os.environ["SUMMARY_API_KEY"],
        base_url=os.environ["SUMMARY_BASE_URL"],
    )
    model_name = os.environ["SUMMARY_MODEL_NAME"]
    for attempt in range(max_retries):
        try:
            # Add a per-request timeout so that OSS summary calls won't hang indefinitely.
            # If the model side is slow or unresponsive, this will raise and be caught below,
            # allowing callers (e.g., visit_web) to log and fall back instead of waiting 360s+.
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.7,
                timeout=300,  # seconds
            )
            msg = chat_response.choices[0].message
            content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
            if content:
                try:
                    json.loads(content)
                except:
                    # extract json from string
                    left = content.find('{')
                    right = content.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        content = content[left:right+1]
                return content
        except Exception as e:
            # Explicitly log timeouts (or other errors) as they happen so callers
            # can see when OSS summary requests are failing due to timeout.
            print(f"[call_server] request timeout on attempt {attempt + 1}/{max_retries}: {e}", flush=True)
            if attempt == (max_retries - 1):
                print(f"[call_server] failed after {max_retries} retries: {e}", flush=True)
                return ""
            continue