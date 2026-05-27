from openai import OpenAI
import importlib
import importlib.util
import argparse
import json
import re
import sys
import os
import asyncio

from context import ContextManager
from resume_context import restore_context_manager

def load_env(env_path):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env(os.path.join(os.path.dirname(__file__), 'config', '.env'))

def load_all_tools(use_meta_tools=False):
    """Load all tools dynamically."""
    tools = {}
    tool_list = [('tool', ['search_web', 'visit_web'])]
    if use_meta_tools:
        tool_list.append(('meta_tool', ['skip', 'rollback', 'snippet', 'compress', 'delete']))
    for tool_type, tool_names in tool_list:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools', tool_type))
        for name in tool_names:
            module = importlib.import_module(name)
            tools[name] = getattr(module, name)
    return tools

def load_tools(dir_path, files):
    schemas = []
    base = os.path.join(os.path.dirname(__file__), dir_path)
    for f in files:
        spec = importlib.util.spec_from_file_location(f, os.path.join(base, f"{f}.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if hasattr(m, 'TOOL_SCHEMA'):
            s = m.TOOL_SCHEMA
            s = json.loads(s) if isinstance(s, str) else s
            schemas.extend(s if isinstance(s, list) else [s])
    return json.dumps(schemas, ensure_ascii=False, indent=2)

def load_prompts(use_meta_tools=False):
    module = importlib.import_module("prompts.prompt")
    prompt_dict = {
        "tool_user_prompt": getattr(module, "standard_tool_user_prompt", None) or getattr(module, "tool_user_prompt", None),
        "tool_user_prompt_last": getattr(module, "standard_tool_user_prompt_last", None) or getattr(module, "tool_user_prompt_last", None),
        "standard_tools": load_tools("tools/tool", ["search_web", "visit_web"]),
        "meta_tools": load_tools("tools/meta_tool", ["skip", "rollback", "snippet", "compress", "delete"]) if use_meta_tools else "",
    }
    return prompt_dict

def extract_tags(text, tag):
    pattern = fr"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        extracted_text = matches[-1].strip()
        return extracted_text
    else:
        return None

def validate_meta_tool_call(llm_response, logger=print):
    """Validate, parse and check meta tool call from LLM response. Returns (str, calls_or_None, error_msg_or_None).
    
    logger: 可注入的日志函数，默认用 print；在多 worker 场景下，可以传入每个 item 自己的 log 函数。
    """
    meta_tool_call_str = extract_tags(llm_response, "meta_tool_call")
    
    if not meta_tool_call_str:
        err = "Error: meta_tool_call tag not found in LLM response"
        logger(err)
        return None, None, err
    
    try:
        parsed = json.loads(meta_tool_call_str)
    except Exception as e:
        err = f"Error: Failed to parse meta_tool_call as JSON: {e}"
        logger(err)
        return meta_tool_call_str, None, err

    # Accept single object or array (one or more meta tools per turn)
    if isinstance(parsed, dict):
        meta_tool_calls = [parsed]
    elif isinstance(parsed, list):
        meta_tool_calls = parsed
    else:
        err = "Error: meta_tool_call must be a JSON object or array"
        print(err)
        return meta_tool_call_str, None, err

    # Validate each meta tool call
    validated_calls = []
    for meta_tool_call in meta_tool_calls:
        # Validate required fields
        if not isinstance(meta_tool_call, dict):
            err = "Error: Each meta tool call must be a JSON object"
            logger(err)
            return meta_tool_call_str, None, err
        
        if "name" not in meta_tool_call:
            err = "Error: meta_tool_call missing 'name' field"
            logger(err)
            return meta_tool_call_str, None, err
        
        meta_tool_name = meta_tool_call["name"]
        if meta_tool_name not in ["skip", "compress", "snippet", "rollback", "delete"]:
            err = f"Error: Unknown meta tool name: {meta_tool_name}"
            logger(err)
            return meta_tool_call_str, None, err
        
        # Ensure arguments field exists (default to empty dict)
        if "arguments" not in meta_tool_call:
            meta_tool_call["arguments"] = {}
        
        validated_calls.append(meta_tool_call)
    
    # Check skip rule: if skip is used, it must be alone
    skip_calls = [call for call in validated_calls if call["name"] == "skip"]
    if skip_calls and len(validated_calls) > 1:
        err = "Error: skip must be used alone and cannot be combined with other meta tools"
        logger(err)
        return meta_tool_call_str, None, err
    # Check rollback rule: if rollback is used, it must be alone
    rollback_calls = [call for call in validated_calls if call["name"] == "rollback"]
    if rollback_calls and len(validated_calls) > 1:
        err = "Error: rollback must be used alone and cannot be combined with other meta tools"
        logger(err)
        return meta_tool_call_str, None, err
    
    return meta_tool_call_str, validated_calls, None

def execute_meta_tool(meta_tool_calls, context_manager, meta_tools, log=None):
    """Execute meta tools with validated meta_tool_calls (list)."""
    # Collect all snippet calls into one list (each elem has step_id, reserve_prefix, reserve_suffix), pass once
    snippet_args_list = []
    for meta_tool_call in meta_tool_calls:
        if meta_tool_call["name"] != "snippet":
            continue
        args = meta_tool_call["arguments"]
        if args.get("step_id") is not None:
            snippet_args_list.append(args)
    if snippet_args_list:
        meta_tools["snippet"](context_manager, snippet_args_list, log=log)
    # Run non-snippet tools after snippet (rollback, compress, delete, skip)
    for meta_tool_call in meta_tool_calls:
        meta_tool_name = meta_tool_call["name"]
        meta_tool_args = meta_tool_call["arguments"]
        if meta_tool_name == "skip":
            pass
        elif meta_tool_name == "rollback":
            meta_tools["rollback"](context_manager, meta_tool_args)
        elif meta_tool_name == "delete":
            meta_tools["delete"](context_manager, meta_tool_args)
        elif meta_tool_name == "compress":
            meta_tools["compress"](context_manager, meta_tool_args)

def validate_standard_tool_call(llm_response):
    """Validate, parse and check standard tool call from LLM response. Returns (str, call_or_None, error_msg_or_None)."""
    standard_tool_call_str = extract_tags(llm_response, "standard_tool_call")
    
    if not standard_tool_call_str:
        return None, None, "Error: standard_tool_call tag not found"
    
    try:
        standard_tool_call = json.loads(standard_tool_call_str)
    except Exception as e:
        return standard_tool_call_str, None, f"Error: Failed to parse standard_tool_call as JSON: {e}"
    
    if "name" not in standard_tool_call:
        return standard_tool_call_str, None, "Error: standard_tool_call missing 'name' field"
    
    if "arguments" not in standard_tool_call:
        return standard_tool_call_str, None, "Error: standard_tool_call missing 'arguments' field"
    
    tool_name = standard_tool_call["name"]
    if tool_name not in ["search_web", "visit_web"]:
        return standard_tool_call_str, None, f"Error: Unknown standard tool name: {tool_name}"
    
    return standard_tool_call_str, standard_tool_call, None

async def execute_standard_tool_async(standard_tool_call, tools):
    """Execute standard tool with validated standard_tool_call."""
    tool_name = standard_tool_call["name"]
    tool_args = standard_tool_call["arguments"]
    
    coro = tools[tool_name](**tool_args)
    try:
        result = await asyncio.wait_for(coro, timeout=480)
    except asyncio.TimeoutError:
        result = f"Error: {tool_name} timed out (480s)."
    return result

async def get_llm_response_async(messages, item_id=None):
    """Async version of get_llm_response."""
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )
    loop = asyncio.get_event_loop()
    response_non_stream = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
    )
    return response_non_stream

async def main_loop_async(args, item, prompt_dict, tools, item_id=None, resume_from_step=None, initial_traj_list=None, log_func=None):
    
    # 简单的 log 函数：默认用 print，可以由外部注入 log_func 定向到各自的 log 文件
    def log(*args, **kwargs):
        kwargs.setdefault("flush", True)
        if log_func is not None:
            log_func(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    if resume_from_step is not None and initial_traj_list is not None:
        # Resume mode: restore context from previous step
        context_manager = resume_from_step
        # tool_count should be the number of completed steps, which is len(initial_traj_list) - 1
        # because initial_traj_list includes the step we're resuming from
        tool_count = len(initial_traj_list) - 1 if len(initial_traj_list) > 0 else 0
        # Save all previous steps except the last one (the resume point step)
        full_traj_list = initial_traj_list[:-1] if len(initial_traj_list) > 0 else []
    else:
        # Normal mode: start fresh
        context_manager = ContextManager()
        tool_count = 0
        full_traj_list = []
    
    def record_traj(messages, output, response):
        """Helper to append one trajectory step into full_traj_list."""
        full_traj_list.append({"user_prompt": messages[-1]["content"]})
        full_traj_list[-1]["reasoning"] = output.choices[0].message.reasoning_content
        full_traj_list[-1]["response"] = response

    id_str = f" (id={item_id})" if item_id else ""
    total_tokens_consumed = 0
    total_prompt_tokens_consumed = 0
    total_completion_tokens_consumed = 0

    while tool_count < args.tool_count_max:
        current_step_rolled_back = False  # reset per step so we only rollback once per step on content_filter
        context_for_messages = context_manager.get_context_for_messages()
        
        # Select prompt based on tool_count
        if tool_count == args.tool_count_max - 1:
            prompt_template = prompt_dict["tool_user_prompt_last"]
        else:
            prompt_template = prompt_dict["tool_user_prompt"]
        
        # Format prompt
        if tool_count == args.tool_count_max - 1:
            messages = [{"role": "user", "content": prompt_template.format(question=item["question"], context=context_for_messages)}]
        elif not args.use_meta_tools:
            messages = [{"role": "user", "content": prompt_template.format(question=item["question"], tools=prompt_dict["standard_tools"], context=context_for_messages)}]
        else:
            messages = [{"role": "user", "content": prompt_template.format(
                question=item["question"],
                standard_tools=prompt_dict["standard_tools"],
                meta_tools=prompt_dict["meta_tools"],
                context=context_for_messages
            )}]
        
        log(f"=== Messages {tool_count + 1}{id_str} ===")
        # log(messages)
        log()
        
        output = None
        for attempt in range(3):
            try:
                output = await asyncio.wait_for(
                    get_llm_response_async(messages, item_id),
                    timeout=720
                )
            except asyncio.TimeoutError:
                log(f"LLM request timed out (id={item_id}, attempt={attempt+1}/3). Retrying...")
                if attempt == 2:
                    print(f"LLM request timed out after all retries (id={item_id}, step={tool_count+1}). Giving up this item.")
                    # give up on this item, record partial trajectory and return
                    full_traj_list.append({
                        "error": "timeout_after_3_attempts",
                        "step": tool_count + 1,
                    })
                    full_traj_list.append({
                        "total_tokens_consumed": total_tokens_consumed,
                        "total_prompt_tokens": total_prompt_tokens_consumed,
                        "total_completion_tokens": total_completion_tokens_consumed,
                    })
                    return full_traj_list
                continue
            except Exception as e:
                msg = str(e)
                if attempt < 2:
                    log(f"LLM transient error (id={item_id}, attempt={attempt+1}/3): {msg}. Retrying...")
                    print(f"LLM transient error (id={item_id}, attempt={attempt+1}/3): {msg}. Retrying...")
                    continue
                print(f"LLM fatal error (id={item_id}): {msg}")
                full_traj_list.append({
                    "error": "llm_exception",
                    "detail": msg,
                    "step": tool_count + 1,
                })
                full_traj_list.append({
                    "total_tokens_consumed": total_tokens_consumed,
                    "total_prompt_tokens": total_prompt_tokens_consumed,
                    "total_completion_tokens": total_completion_tokens_consumed,
                })
                return full_traj_list
            # output = await get_llm_response_async_local(messages, item_id)
            # output = await get_llm_response_async_dpsk(messages, item_id)
            usage = getattr(output, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                tt = getattr(usage, "total_tokens", None)
                if pt is not None:
                    total_prompt_tokens_consumed += pt
                if ct is not None:
                    total_completion_tokens_consumed += ct
                if tt is not None:
                    total_tokens_consumed += tt
                if pt is not None or ct is not None or tt is not None:
                    log(f"总token id={item_id} (本回合: prompt={pt}, completion={ct}, total={tt} | 累计: prompt={total_prompt_tokens_consumed}, completion={total_completion_tokens_consumed}, total={total_tokens_consumed})")
            log(f"=== LLM Response {tool_count + 1}{id_str} ===")
            # log(output)
            log()

            # Check for content_filter
            if output.choices[0].finish_reason == 'content_filter':
                # Try to rollback once per step if possible
                if tool_count > 0 and not current_step_rolled_back:
                    log(f"Content filter detected (id={item_id}). Rolling back one step and retrying...")
                    # Remove last step from context (key is (tool_count, tool_count) per ContextManager)
                    last_step_key = (tool_count, tool_count)
                    context_manager.delete(last_step_key)
                    # Remove last trajectory entry
                    if full_traj_list:
                        full_traj_list.pop()
                    # Rollback tool_count
                    tool_count -= 1
                    current_step_rolled_back = True
                    # Break out of attempt loop and continue while loop to retry
                    break
                else:
                    log(f"Content filter detected (id={item_id}). Recording trajectory and exiting.")
                    full_traj_list.append({"content_filter": True})
                    full_traj_list.append({
                        "total_tokens_consumed": total_tokens_consumed,
                        "total_prompt_tokens": total_prompt_tokens_consumed,
                        "total_completion_tokens": total_completion_tokens_consumed,
                    })
                    return full_traj_list

            response = output.choices[0].message.content
            
            if '<answer>' in response and '</answer>' in response:
                # Only treat as final answer if </answer> is the end of the message (ignoring whitespace).
                answer_end_idx = response.rfind("</answer>")
                tail = response[answer_end_idx + len("</answer>"):] if answer_end_idx != -1 else ""
                if tail.strip() == "":
                    # Final answer step: record trajectory and stop
                    record_traj(messages, output, response)
                    full_traj_list.append({
                        "total_tokens_consumed": total_tokens_consumed,
                        "total_prompt_tokens": total_prompt_tokens_consumed,
                        "total_completion_tokens": total_completion_tokens_consumed,
                    })
                    return full_traj_list
            
            motivation = extract_tags(response, "motivation")
            if args.use_meta_tools:
                # 将 meta_tool 的校验错误也写入当前 item 的 log（通过 logger 注入）
                meta_tool_call_str, meta_tool_calls, meta_err = validate_meta_tool_call(response, logger=log)
                if meta_tool_calls is None:
                    log(f"Error: Invalid meta tool call format. Retrying for the {attempt+1}th time.")
                    if attempt == 2:
                        next_step = context_manager.get_step_max() + 1
                        context_manager.update_context_dict_list(motivation, meta_tool_call_str, meta_err or "Error: invalid meta tool call format", next_step)
                        record_traj(messages, output, response)
                        tool_count += 1
                        break
                    continue
            else:
                meta_tool_calls = []
            # Validate standard tool call
            standard_tool_call_str, standard_tool_call, std_err = validate_standard_tool_call(response)
            if standard_tool_call is None:
                log(f"Error: Invalid standard tool call format: {std_err}. Retrying for the {attempt+1}th time.")
                if attempt == 2:
                    next_step = context_manager.get_step_max() + 1
                    context_manager.update_context_dict_list(motivation, standard_tool_call_str, std_err or "Error: invalid tool call format", next_step)
                    record_traj(messages, output, response)
                    tool_count += 1
                    break
                continue
            
            # If both validations passed, execute standard tool
            for tool_attempt in range(3):
                standard_tool_response = await execute_standard_tool_async(standard_tool_call, tools)
                if "timed out" not in standard_tool_response:
                    break
                log(f"Tool timed out (id={item_id}): retrying tool {standard_tool_call['name']} ({tool_attempt+1}).")
                print(f"Tool timed out (id={item_id}): retrying tool {standard_tool_call['name']} ({tool_attempt+1}).")
            
            # Check if standard tool execution resulted in an error
            if standard_tool_response.startswith("Error:"):
                log(f"Error during standard tool execution: {standard_tool_response}. Retrying for the {attempt+1}th time.")
                print(f"Error during standard tool execution: {standard_tool_response}. Retrying for the {attempt+1}th time.")

                if attempt == 2:
                    # Skip this item: record error and return
                    log(f"Final failure (id={item_id}): skip item due to standard tool error: {standard_tool_response}")
                    print(f"Final failure (id={item_id}): skip item due to standard tool error: {standard_tool_response}")
                    next_step = context_manager.get_step_max() + 1
                    context_manager.update_context_dict_list(motivation, standard_tool_call_str, standard_tool_response, next_step)
                    record_traj(messages, output, response)
                    full_traj_list[-1]["error"] = standard_tool_response
                    full_traj_list.append({"total_tokens_consumed": total_tokens_consumed, "total_prompt_tokens": total_prompt_tokens_consumed, "total_completion_tokens": total_completion_tokens_consumed})
                    return full_traj_list
                continue

            # Update context with standard tool result (after meta tool: step numbers are continuous from current context)
            meta_tools_dict = {"compress": tools.get("compress"), "skip": tools.get("skip"), "snippet": tools.get("snippet"), "rollback": tools.get("rollback"), "delete": tools.get("delete")}
            execute_meta_tool(meta_tool_calls, context_manager, meta_tools_dict, log=log)

            next_step = context_manager.get_step_max() + 1
            context_manager.update_context_dict_list(motivation, standard_tool_call_str, standard_tool_response, next_step)
            
            # Record trajectory and increment tool_count on success
            record_traj(messages, output, response)
            
            tool_count += 1
            break

    full_traj_list.append({
        "total_tokens_consumed": total_tokens_consumed,
        "total_prompt_tokens": total_prompt_tokens_consumed,
        "total_completion_tokens": total_completion_tokens_consumed,
    })
    return full_traj_list

async def process_item(args, item_data, idx, prompt_dict, tools, result_dir):
    """Process a single item asynchronously."""
    item = {"question": item_data["query"]}
    item_id = item_data.get("id", str(idx))
    
    result_path = os.path.join(result_dir, f"result_{item_id}.json")
    log_dir = os.path.join(result_dir, "logs")
    log_path = os.path.join(log_dir, f"{item_id}.log")

    # Skip without creating log when result already exists
    if args.resume_from_step is None and os.path.exists(result_path):
        print(f"Skipping item {item_id} (result already exists)")
        return

    os.makedirs(log_dir, exist_ok=True)
    # 将该 item 的大部分日志写入单独的 log 文件（核心状态提示仍输出到全局 stdout）
    with open(log_path, "a", encoding="utf-8") as lf:
        def log_to_file(*args, **kwargs):
            kwargs.setdefault("flush", True)
            print(*args, file=lf, **kwargs)
        # Resume mode: restore from existing file in original folder
        if args.resume_from_step is not None:
            # Get original folder name (remove "_resumed" suffix if present)
            suffix = "_meta_react_separate" if args.use_meta_tools else "_react_separate"
            original_result_folder_name = f"{os.path.splitext(os.path.basename(args.dataset))[0]}{suffix}"
            original_result_dir = os.path.join(os.path.dirname(__file__), "result", original_result_folder_name)
            original_result_path = os.path.join(original_result_dir, f"result_{item_id}.json")
            
            if not os.path.exists(original_result_path):
                print(f"Warning: Resume requested but result file not found for item {item_id} at {original_result_path}")
                return
            try:
                with open(original_result_path, 'r', encoding='utf-8') as f:
                    existing_traj = json.load(f)
                if len(existing_traj) <= args.resume_from_step:
                    print(f"Warning: Step {args.resume_from_step} not found in result file for item {item_id}")
                    return
                # Restore context from the specified step (read from original file)
                context_manager = restore_context_manager(original_result_path, args.resume_from_step)
                # Get existing trajectory up to resume point
                initial_traj = existing_traj[:args.resume_from_step + 1]
                # 这里的提示信息走全局 stdout
                print(f"Resuming item {item_id} from step {args.resume_from_step + 1}")
                full_traj_list = await main_loop_async(
                    args,
                    item,
                    prompt_dict,
                    tools,
                    item_id,
                    resume_from_step=context_manager,
                    initial_traj_list=initial_traj,
                    log_func=log_to_file,
                )
                # Save to new folder (result_dir already points to the _resumed folder)
                with open(result_path, "w", encoding="utf-8") as rf:
                    json.dump(full_traj_list, rf, ensure_ascii=False, indent=2)
                print(f"Updated result for item {item_id} in resumed folder")
                return
            except Exception as e:
                print(f"Error resuming item {item_id}: {e}")
                return

        try:
            full_traj_list = await main_loop_async(args, item, prompt_dict, tools, item_id, log_func=log_to_file)
        except asyncio.TimeoutError:
            # Avoid crashing the whole run if a single item's LLM calls
            # keep timing out even after internal retries.
            print(f"Error: item {item_id} timed out after multiple LLM retries. Skipping this item.")
            return
       # Save trajectory for this item using id
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(full_traj_list, rf, ensure_ascii=False, indent=2)
        log_to_file(f"Saved result for item {item_id}")
        print(f"Saved result for item {item_id}")

async def main_async(args):
    """Main async function to process dataset with concurrency."""
    tools = load_all_tools(args.use_meta_tools)
    prompt_dict = load_prompts(args.use_meta_tools)
    
    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Filter dataset by item_ids if specified
    if args.item_ids:
        item_ids_set = set(args.item_ids)
        dataset = [item for item in dataset if item.get("id") in item_ids_set]
        if not dataset:
            print(f"Warning: No items found with the specified IDs: {args.item_ids}")
            return
        print(f"Filtering dataset to {len(dataset)} items with IDs: {args.item_ids}")
    
    # Extract dataset name from path (e.g., "dataset/browsecomp_test5.json" -> "browsecomp_test5")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    result_folder_name = f"{dataset_name}_{'meta_' if args.use_meta_tools else ''}react_separate"
    
    # If resuming, use a new folder with "_resumed" suffix
    if args.resume_from_step is not None:
        result_folder_name = f"{result_folder_name}_resumed"
    
    # Ensure result directory exists
    result_dir = os.path.join(os.path.dirname(__file__), "result", result_folder_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Process items with concurrency control
    num_workers = args.num_workers
    semaphore = asyncio.Semaphore(num_workers)
    
    async def process_with_semaphore(idx, item_data):
        async with semaphore:
            await process_item(args, item_data, idx, prompt_dict, tools, result_dir)
        
    # Create tasks for all items
    tasks = [process_with_semaphore(idx, item_data) for idx, item_data in enumerate(dataset)]
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    print(f"All {len(dataset)} items processed.")
    print(f"Exiting...")
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_count_max", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="dataset/browsecomp_test1.json")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--item_ids", type=str, nargs="+", default=None, help="List of item IDs to process (only process these specific items)")
    parser.add_argument("--resume_from_step", type=int, default=None, help="Resume from step N (0-based, e.g., 59 for step 60)")
    parser.add_argument("--use_meta_tools", type=lambda x: x.lower() == "true", default=True, help="true or false")
    args = parser.parse_args()

    asyncio.run(main_async(args))