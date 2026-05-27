TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "skip",
            "description": "Skip this step, do nothing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def skip() -> str:
    """Skip this step, do nothing."""
    return "Skipped."

