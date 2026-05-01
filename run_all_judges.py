#!/usr/bin/env python3
"""Run all remaining judge evaluations sequentially.

Runs local Ollama judges first, then the GPT-o3-mini comparison via OpenRouter.
Supports resume — skips files already evaluated by each judge.

Usage:
    python3 run_all_judges.py
"""
import os
import sys
import time
import glob
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://klusai-macstudio.pike-albacore.ts.net:11434/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

FABLE_FILES = sorted([
    "data/fables/aya-23-8b-ngn/tf_fables_aya-23-8b-ngn_dt250402-085836.jsonl",
    "data/fables/deepseek-llm-7b-chat-gzs/tf_fables_deepseek-llm-7b-chat-gzs_dt250402-091410.jsonl",
    "data/fables/falcon3-7b-instruct-ohd/tf_fables_falcon3-7b-instruct-ohd_dt250402-085836.jsonl",
    "data/fables/llama-3-1-8b-instruct-mpp/tf_fables_llama-3-1-8b-instruct-mpp_dt250402-091028.jsonl",
    "data/fables/llama-3-1-tulu-3-8b-eoz/tf_fables_llama-3-1-tulu-3-8b-eoz_dt250402-085836.jsonl",
    "data/fables/llama-3-2-1b-instruct-sei/tf_fables_llama-3-2-1b-instruct-sei_dt250402-091602.jsonl",
    "data/fables/mistral-7b-instruct-v0-3-qqj/tf_fables_mistral-7b-instruct-v0-3-qqj_dt250402-091028.jsonl",
    "data/fables/phi-3-mini-4k-instruct-bma/tf_fables_phi-3-mini-4k-instruct-bma_dt250402-090453.jsonl",
    "data/fables/qwen2-5-7b-instruct-rjv/tf_fables_qwen2-5-7b-instruct-rjv_dt250402-085836.jsonl",
    "data/fables/smollm2-1-7b-instruct-ins/tf_fables_smollm2-1-7b-instruct-ins_dt250402-091028.jsonl",
])

JUDGES = [
    {
        "model": "exaone3.5:32b",
        "base_url": OLLAMA_URL,
        "api_key": "ollama",
        "env": {
            "EVAL_STRICT_SCHEMA": "1",
            "EVAL_NO_THINK": "0",
            "EVAL_MAX_TOKENS": "4096",
        },
    },
    {
        "model": "granite3.3:8b",
        "base_url": OLLAMA_URL,
        "api_key": "ollama",
        "env": {
            "EVAL_STRICT_SCHEMA": "1",
            "EVAL_NO_THINK": "0",
            "EVAL_MAX_TOKENS": "4096",
        },
    },
    {
        "model": "openai/o4-mini",
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
        "env": {
            "EVAL_STRICT_SCHEMA": "0",
            "EVAL_NO_THINK": "0",
            "EVAL_MAX_TOKENS": "4096",
            "EVAL_TEMPERATURE": "1",
        },
    },
]

OUTPUT_DIR = "data/evaluations"


def model_safe(name: str) -> str:
    return name.replace("/", "-").replace(":", "-")


def already_evaluated(model: str) -> set:
    """Return set of fable basenames already evaluated by this model."""
    safe = model_safe(model)
    done = set()
    for path in glob.glob(os.path.join(OUTPUT_DIR, f"*{safe}*.jsonl")):
        basename = os.path.basename(path)
        for ff in FABLE_FILES:
            fb = os.path.splitext(os.path.basename(ff))[0]
            if fb in basename:
                done.add(ff)
    return done


def run_judge(judge_cfg: dict) -> None:
    model = judge_cfg["model"]
    env_overrides = judge_cfg["env"]

    os.environ["EVAL_BASE_URL"] = judge_cfg["base_url"]
    os.environ["EVAL_API_KEY"] = judge_cfg["api_key"]
    os.environ["EVAL_MODEL"] = model
    for k, v in env_overrides.items():
        os.environ[k] = v

    import importlib
    import tinyfabulist.evaluate.en as en_mod
    import tinyfabulist.evaluate.utils as utils_mod
    importlib.reload(utils_mod)
    importlib.reload(en_mod)

    done = already_evaluated(model)
    remaining = [f for f in FABLE_FILES if f not in done]

    print(f"\n{'='*60}")
    print(f"JUDGE: {model} | Already done: {len(done)} | Remaining: {len(remaining)}")
    print(f"{'='*60}")

    if not remaining:
        print(f"All files already evaluated by {model}. Skipping.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, fpath in enumerate(remaining):
        t0 = time.time()
        fname = os.path.basename(fpath)
        print(f"\n--- [{i+1}/{len(remaining)}] {fname} with {model} ---")
        print(f"    Started: {datetime.now().strftime('%H:%M:%S')}")

        try:
            en_mod.evaluate_file(fpath, output_dir=OUTPUT_DIR)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        elapsed = time.time() - t0
        print(f"    Finished in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    print(f"\n{'='*60}")
    print(f"JUDGE {model} COMPLETE")
    print(f"{'='*60}")


def main():
    total_t0 = time.time()
    print(f"Starting all judge evaluations at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Judges: {[j['model'] for j in JUDGES]}")
    print(f"Fable files: {len(FABLE_FILES)}")

    for judge in JUDGES:
        run_judge(judge)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"ALL JUDGES COMPLETE in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
