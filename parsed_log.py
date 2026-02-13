import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


# -----------------------------
# Regex helpers for tag parsing
# -----------------------------
TAG_RE = {
    "think": re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL),
    "search": re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL),
    "answer": re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL),
    "information": re.compile(r"<information>\s*(.*?)\s*</information>", re.DOTALL),
}

Q_RE = re.compile(r"Your question:\s*(.*?)(?:\n|$)")

HISTORY_BLOCK_RE = re.compile(
    r"History:\s*(.*)\n\n\nNow it's your turn to respond for the current step\.",
    re.DOTALL
)

STEP_RE = re.compile(r"Step\s+(\d+)\s*:\s*", re.DOTALL)

# Parse "Doc 1: xxx\n....\nDoc 2: yyy\n...." blocks
DOC_SPLIT_RE = re.compile(r"(Doc\s+\d+:\s*)(.*?)(?=(?:\nDoc\s+\d+:)|\Z)", re.DOTALL)


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1) if m else None


def _all_matches(pattern: re.Pattern, text: str) -> List[str]:
    return [m.group(1).strip() for m in pattern.finditer(text)]


# -----------------------------------
# Parsing model response: think/search/answer
# -----------------------------------
def parse_response_fields(response: str) -> Dict[str, Any]:
    think = _first_match(TAG_RE["think"], response)
    searches = _all_matches(TAG_RE["search"], response)
    answers = _all_matches(TAG_RE["answer"], response)
    return {
        "think": think.strip() if think else None,
        "search_queries": searches,
        "answers": answers,
        "raw_response": response,
    }


# -----------------------------------
# Parsing prompt history: question + prior searches/information
# -----------------------------------
def parse_docs_from_information_result(result_text: str) -> List[Dict[str, str]]:
    """
    Split information_json["result"] into a list of docs.
    Assumes it contains "Doc N: ..." blocks.
    """
    docs = []
    for _, body in DOC_SPLIT_RE.findall((result_text or "").strip()):
        body = body.strip()
        lines = body.splitlines()
        title = lines[0].strip() if lines else ""
        content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        docs.append({"title": title, "content": content})
    return docs


def split_history_steps(history_text: str) -> List[Dict[str, Any]]:
    """
    Turn the History block into list[{"step": int, "queries": [...], "information_json": {...}, "docs": [...]}]
    """
    step_positions = [(m.start(), int(m.group(1))) for m in STEP_RE.finditer(history_text)]
    if not step_positions:
        return []

    steps = []
    for i, (pos, step_no) in enumerate(step_positions):
        end = step_positions[i + 1][0] if i + 1 < len(step_positions) else len(history_text)
        chunk = history_text[pos:end]

        queries = _all_matches(TAG_RE["search"], chunk)
        info_raw = _first_match(TAG_RE["information"], chunk)

        info_json = None
        if info_raw:
            # Often a JSON string like {"result": "..."}
            try:
                info_json = json.loads(info_raw)
            except Exception:
                info_json = None

        step_item = {
            "step": step_no,
            "queries": queries,
            "information_raw": info_raw,
            "information_json": info_json,
        }

        if isinstance(info_json, dict) and isinstance(info_json.get("result"), str):
            step_item["docs"] = parse_docs_from_information_result(info_json["result"])

        steps.append(step_item)

    return steps


def parse_prompt_fields(prompt: str) -> Dict[str, Any]:
    question = _first_match(Q_RE, prompt)

    history = []
    hb = _first_match(HISTORY_BLOCK_RE, prompt)
    if hb:
        history = split_history_steps(hb)

    return {
        "question": question.strip() if question else None,
        "history": history,
        "raw_prompt": prompt,
    }


# -----------------------------------
# Main JSONL parsing + enrichment
# -----------------------------------
def parse_jsonl(input_path: str, output_path: str):
    out = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Line {line_no} JSON decode error: {e}\nFirst 200 chars: {line[:200]}"
                )

            step_item = {
                "step": obj.get("step"),
                "timestamp": obj.get("timestamp"),
                "batch_size": obj.get("batch_size"),
                "active_count": obj.get("active_count"),
                "generation_stats": obj.get("generation_stats", {}),
                "rewards": obj.get("rewards", {}),
                "episode_stats": obj.get("episode_stats", {}),
                # step-level tool calling summary
                "tool_calling_summary": {},
                # samples (enriched)
                "samples": []
            }

            # ---- Step-level tool_calling summary ----
            tc = obj.get("tool_calling", {}) or {}
            details = tc.get("details", []) or []

            step_item["tool_calling_summary"] = {
                "total_tool_calls": tc.get("total_tool_calls"),
                "current_step_tool_calls": tc.get("current_step_tool_calls"),
                "by_tool_name": dict(
                    Counter(d.get("tool_name", "UNKNOWN") for d in details if d.get("tool_calling"))
                ),
                "valid_calls": sum(
                    1 for d in details if d.get("tool_calling") and d.get("is_action_valid") == 1
                ),
                "invalid_calls": sum(
                    1 for d in details if d.get("tool_calling") and d.get("is_action_valid") == 0
                ),
                "not_tool_calling": sum(1 for d in details if not d.get("tool_calling")),
            }

            # Group step details by sample_idx
            per_sample_tc = defaultdict(list)
            for d in details:
                per_sample_tc[d.get("sample_idx")].append(d)

            # ---- Samples ----
            for s in obj.get("samples", []) or []:
                sample_idx = s.get("sample_idx")
                s_tc = s.get("tool_calling", {}) or {}
                sample_details = per_sample_tc.get(sample_idx, [])

                prompt = s.get("prompt") or ""
                response = s.get("response") or ""

                sample_item = {
                    "sample_idx": sample_idx,
                    "active": s.get("active"),
                    "prompt": prompt,
                    "response": response,
                    "reward": s.get("reward"),
                    "episode_reward": s.get("episode_reward"),
                    "episode_length": s.get("episode_length"),
                    "done": s.get("done"),

                    # original sample tool_calling field
                    "tool_calling": {
                        "tool_calling": s_tc.get("tool_calling"),
                        "is_action_valid": s_tc.get("is_action_valid"),
                        "tool_name": s_tc.get("tool_name"),
                    },

                    # aggregated from step details (more complete)
                    "tool_calling_from_step_details": {
                        "num_calls": sum(1 for d in sample_details if d.get("tool_calling")),
                        "valid_calls": sum(
                            1 for d in sample_details if d.get("tool_calling") and d.get("is_action_valid") == 1
                        ),
                        "invalid_calls": sum(
                            1 for d in sample_details if d.get("tool_calling") and d.get("is_action_valid") == 0
                        ),
                        "tools": dict(
                            Counter(d.get("tool_name", "UNKNOWN") for d in sample_details if d.get("tool_calling"))
                        ),
                        # If you want raw details (can be huge), uncomment:
                        # "details": sample_details,
                    },

                    # NEW: parsed prompt + response
                    "parsed_prompt": parse_prompt_fields(prompt),
                    "parsed_response": parse_response_fields(response),
                }

                step_item["samples"].append(sample_item)

            out.append(step_item)

    with open(output_path, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parse_jsonl("rollout_details_20260212_082230.jsonl", "parsed.json")
    print("Wrote parsed.json")
