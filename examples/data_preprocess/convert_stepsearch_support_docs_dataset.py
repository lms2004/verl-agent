#!/usr/bin/env python3
# Copyright 2026
#
# Convert a StepSearch-style raw dataset into verl-agent search training format.
# This converter focuses on support_docs-based step reward adaptation.

import argparse
import json
import os
import random
from typing import Any

import pandas as pd


DEFAULT_PROMPT_TEMPLATE = (
    "## Background\n"
    "You are a deep AI research assistant with a search tool.\n"
    "You can think and call search when needed.\n\n"
    "## Response format\n"
    "Use this loop when needed: <plan>...</plan> <search>...</search> <information>...</information> <observation>...</observation>\n"
    "When ready, answer with <answer>...</answer>.\n\n"
    "Question: {question}\n"
)


def read_input_file(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if lower.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported input format for {path}. Use .parquet/.jsonl/.json")


def normalize_support_docs(raw_docs: Any) -> list[dict]:
    """Normalize raw support docs into list[{'title', 'paragraph_text'}]."""
    if raw_docs is None:
        return []
    if not isinstance(raw_docs, list):
        return []

    docs: list[dict] = []

    def _append_if_valid(item: Any):
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            paragraph_text = str(item.get("paragraph_text", "")).strip()
            if title or paragraph_text:
                docs.append({"title": title, "paragraph_text": paragraph_text})

    for item in raw_docs:
        if isinstance(item, list):
            for sub_item in item:
                _append_if_valid(sub_item)
        else:
            _append_if_valid(item)

    return docs


def build_target_list(final_answer: Any, answer_aliases: Any) -> list[str]:
    candidates: list[str] = []
    if final_answer is not None:
        ans = str(final_answer).strip()
        if ans:
            candidates.append(ans)

    if isinstance(answer_aliases, list):
        for alias in answer_aliases:
            if alias is None:
                continue
            alias_s = str(alias).strip()
            if alias_s:
                candidates.append(alias_s)

    # de-duplicate while preserving order
    uniq: list[str] = []
    seen = set()
    for x in candidates:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def process_row(
    row: pd.Series,
    split_name: str,
    row_index: int,
    data_source: str,
    prompt_template: str,
) -> dict:
    question = str(row.get("question", "")).strip()
    if question and not question.endswith("?"):
        question += "?"

    support_docs = normalize_support_docs(row.get("sub_support_docs", []))
    target = build_target_list(row.get("final_answer", ""), row.get("answer_aliases", []))

    # Keep a single-answer fallback to avoid empty targets.
    if not target:
        target = [""]

    ground_truth = {"target": target}
    prompt = [{"role": "user", "content": prompt_template.format(question=question)}]

    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split_name,
            "index": int(row.get("id", row_index)),
            "question": question,
            "support_docs": support_docs,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
        "env_kwargs": {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": data_source,
            "support_docs": support_docs,
        },
        "metadata": {
            "raw_id": row.get("id", row_index),
            "hops": row.get("hops", None),
            "sub_questions": row.get("sub_questions", []),
            "sub_answers": row.get("sub_answers", []),
            "sub_searchs": row.get("sub_searchs", []),
        },
    }


def sample_df(df: pd.DataFrame, sample_num: int | None, seed: int) -> pd.DataFrame:
    if sample_num is None:
        return df
    if sample_num <= 0:
        return df.iloc[0:0].copy()
    if sample_num >= len(df):
        return df
    return df.sample(n=sample_num, random_state=seed).reset_index(drop=True)


def split_df(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    idx = list(range(len(df)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    test_size = int(len(df) * test_ratio)
    test_idx = set(idx[:test_size])

    train_rows = [i for i in range(len(df)) if i not in test_idx]
    test_rows = [i for i in range(len(df)) if i in test_idx]
    return df.iloc[train_rows].reset_index(drop=True), df.iloc[test_rows].reset_index(drop=True)


def convert_split(
    df_raw: pd.DataFrame,
    split_name: str,
    data_source: str,
    prompt_template: str,
) -> pd.DataFrame:
    rows = [
        process_row(
            row=df_raw.iloc[i],
            split_name=split_name,
            row_index=i,
            data_source=data_source,
            prompt_template=prompt_template,
        )
        for i in range(len(df_raw))
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert StepSearch-like raw data to verl-agent search parquet format."
    )
    parser.add_argument("--input_path", type=str, default=None, help="Single input file path (.parquet/.jsonl/.json).")
    parser.add_argument("--train_input_path", type=str, default=None, help="Train input file path.")
    parser.add_argument("--test_input_path", type=str, default=None, help="Test input file path.")
    parser.add_argument("--output_dir", type=str, default="./data/stepsearch_support_docs_processed")
    parser.add_argument("--data_source", type=str, default="musi")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Used only when --input_path is provided.")
    parser.add_argument("--train_sample_num", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--test_sample_num", type=int, default=None, help="Optional test subset size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_template_path", type=str, default=None, help="Optional text file for prompt template.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompt_template_path:
        with open(args.prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    if args.input_path:
        df_all = read_input_file(args.input_path)
        train_raw, test_raw = split_df(df_all, test_ratio=args.test_ratio, seed=args.seed)
    else:
        if not args.train_input_path or not args.test_input_path:
            raise ValueError("Use either --input_path OR both --train_input_path and --test_input_path.")
        train_raw = read_input_file(args.train_input_path)
        test_raw = read_input_file(args.test_input_path)

    train_raw = sample_df(train_raw, args.train_sample_num, args.seed)
    test_raw = sample_df(test_raw, args.test_sample_num, args.seed)

    train_out = convert_split(train_raw, "train", args.data_source, prompt_template)
    test_out = convert_split(test_raw, "test", args.data_source, prompt_template)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)

    print(
        json.dumps(
            {
                "train_rows": len(train_out),
                "test_rows": len(test_out),
                "train_path": train_path,
                "test_path": test_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

