"""Resume context from a saved result file."""
import json
import re
from context import ContextManager

def parse_context_from_prompt(prompt_text):
    """
    Parse context (steps) from prompt text.

    Returns:
        step_dict: Dictionary with tuple keys (start, end) and content values
    """
    step_dict = {}

    step_match = re.search(
        r'### Previous Steps\s*\n\n(.*)',
        prompt_text,
        re.DOTALL
    )

    if not step_match:
        return step_dict

    step_content = step_match.group(1)

    if step_content.strip() == "EMPTY":
        return step_dict

    header_pattern = r'\*\*\[(?:Compressed )?Step (\d+)(?:~(\d+))?\]\*\*'

    headers = list(re.finditer(header_pattern, step_content))

    for i, match in enumerate(headers):
        start_step = int(match.group(1))
        end_str = match.group(2)
        end_step = start_step if end_str is None else int(end_str)

        start_pos = match.end()

        if i + 1 < len(headers):
            end_pos = headers[i + 1].start()
        else:
            end_pos = len(step_content)

        content = step_content[start_pos:end_pos].strip()

        step_dict[(start_step, end_step)] = {
            "content": content
        }

    return step_dict


def restore_context_manager(result_path, step_index=59):
    """
    Restore ContextManager from a result file at a specific step.
    """
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    if step_index >= len(result_data):
        raise ValueError(
            f"Step index {step_index} is out of range "
            f"(max: {len(result_data)-1})"
        )

    step_data = result_data[step_index]
    steps_text = step_data.get("user_prompt", "")

    step_dict = parse_context_from_prompt(steps_text)

    context_manager = ContextManager()
    context_manager.step_dict = step_dict

    return context_manager
