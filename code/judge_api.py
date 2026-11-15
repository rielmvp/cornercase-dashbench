#!/usr/bin/env python3
"""
judge_api.py: judge DashBench _for_judging.json files with DeepSeek-V3 (the judge of the paper).

Input : a single *_for_judging.json file (from convert_to_judging_format.py) or a folder of them
Output: judge/ folder next to the for_judging/ folder; each item gets the judge's reply and a format_classification
        (starts_yes / starts_no / ...). Accuracy is the share of starts_yes. Re-running resumes unfinished items.

Your own OpenRouter key goes in api_key/openrouter_key.txt (or pass --api-key-path).

Examples
--------
python code/judge_api.py results/full/gpt-5-2025-08-07/for_judging/full_gpt-5-2025-08-07_for_judging.json
python code/judge_api.py results/full/gpt-5-2025-08-07/for_judging/ --dry-run
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ── Thread-local API client ───────────────────────────────────────────────────

thread_local = threading.local()

STRICT_YESNO_SYSTEM = (
    "You are a strict binary judge. Your response must start with exactly Yes or No. "
    "Do not use markdown. Do not write analysis before the verdict. After the first "
    "word, add a short explanation."
)

YESNO_SUFFIX = (
    "\n\nOutput format requirement:\n"
    "Start your response with exactly one of these two words: Yes or No.\n"
    "Do not put markdown, bold markers, headings, or analysis before the verdict.\n"
    "After the verdict, provide a concise explanation."
)


# ── API client ────────────────────────────────────────────────────────────────

def get_client(api_key, provider, referer, app_title):
    key = (provider, api_key, referer, app_title)
    if not hasattr(thread_local, "client") or getattr(thread_local, "client_key", None) != key:
        if provider == "openrouter":
            thread_local.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": referer or "https://localhost",
                    "X-Title": app_title or "DashBench Judge",
                },
            )
        elif provider == "openai":
            thread_local.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        thread_local.client_key = key
    return thread_local.client


def call_api(api_key, provider, model, prompt, temperature, max_tokens,
             referer, app_title, strategy):
    client = get_client(api_key, provider, referer, app_title)

    if strategy == "yesno_system_only":
        messages = [
            {"role": "system", "content": STRICT_YESNO_SYSTEM},
            {"role": "user",   "content": [{"type": "text", "text": prompt}]},
        ]
    elif strategy == "yesno_system":
        messages = [
            {"role": "system", "content": STRICT_YESNO_SYSTEM},
            {"role": "user",   "content": [{"type": "text", "text": prompt + YESNO_SUFFIX}]},
        ]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    params = {"model": model, "messages": messages}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    for attempt, wait in enumerate([0, 15, 30, 60, None]):
        try:
            resp = client.chat.completions.create(**params).model_dump()
            if resp.get("error"):
                raise RuntimeError(f"API error: {resp['error']}")
            return resp
        except Exception as exc:
            print(f"  Attempt {attempt + 1} failed: {exc}", flush=True)
            if wait is None:
                raise
            if wait:
                time.sleep(wait)


# ── Response classification ───────────────────────────────────────────────────

def classify(response):
    try:
        text = (response["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return "missing_content"
    if not text:
        return "empty"
    if text.startswith("Yes"):
        return "starts_yes"
    if text.startswith("No"):
        return "starts_no"
    try:
        v = json.loads(text).get("verdict")
        return "json_yes" if v == "Yes" else ("json_no" if v == "No" else "json_other")
    except Exception:
        return "other"


def is_good(cls):
    return cls in {"starts_yes", "starts_no", "json_yes", "json_no"}


def result_is_good(item):
    r = item.get("judge_response")
    return bool(r) and is_good(classify(r))


# ── Single item ───────────────────────────────────────────────────────────────

def process_item(item, args, api_key, max_tokens):
    for attempt in range(2):
        resp = call_api(
            api_key, args.provider, args.model,
            item["prompt"], args.temperature, max_tokens,
            args.referer, args.app_title, args.strategy,
        )
        cls = classify(resp)
        out = {**item, "judge_response": resp, "format_classification": cls}
        print(f"  {item['ID']} -> {cls}", flush=True)
        if is_good(cls):
            return out
        if attempt == 0:
            print(f"  Bad format, retrying in {args.bad_format_retry_wait}s …", flush=True)
            time.sleep(args.bad_format_retry_wait)
    return out


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(items, args, api_key, max_tokens, concurrency, label,
              results_by_id=None, save_fn=None, save_every=200):
    print(f"\n  [{label}] {len(items)} items  concurrency={concurrency}  max_tokens={max_tokens}",
          flush=True)
    if not items:
        return [], []
    successes, failed = [], []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(process_item, it, args, api_key, max_tokens): it for it in items}
        for f in as_completed(futs):
            it = futs[f]
            try:
                result = f.result()
                successes.append(result)
                if results_by_id is not None:
                    results_by_id[result["ID"]] = result
            except Exception as e:
                print(f"  FAILED {it['ID']}: {e}", flush=True)
                failed.append(it["ID"])
            completed += 1
            if save_fn and completed % save_every == 0:
                save_fn()
    return successes, failed


# ── One file ──────────────────────────────────────────────────────────────────

def judge_file(input_path: str, output_path: str, args, api_key):
    print(f"\n{'='*60}", flush=True)
    print(f"INPUT : {input_path}", flush=True)
    print(f"OUTPUT: {output_path}", flush=True)

    with open(input_path) as f:
        data = json.load(f)

    # Load existing results (resume support)
    existing = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)

    results_by_id = {it["ID"]: it for it in existing if "ID" in it}
    good_ids = {iid for iid, it in results_by_id.items() if result_is_good(it)}

    items_to_do = [it for it in data if it["ID"] not in good_ids]
    print(f"  Total: {len(data)}  Already judged: {len(good_ids)}  To do: {len(items_to_do)}",
          flush=True)

    if not items_to_do:
        print("  All items already have clean judgments — skipping.", flush=True)
        return

    def save():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ordered = [results_by_id[it["ID"]] for it in data if it["ID"] in results_by_id]
        remaining = [it for iid, it in results_by_id.items()
                     if iid not in {x["ID"] for x in data}]
        with open(output_path, "w") as f:
            json.dump(ordered + remaining, f, indent=2)

    phases = [
        ("Phase 1 — normal",         args.concurrency,     args.max_tokens),
        ("Phase 2 — low concurrency", args.low_concurrency, args.max_tokens),
        ("Phase 3 — more tokens",     args.low_concurrency, args.retry_max_tokens),
    ]

    remaining = items_to_do
    for label, concurrency, max_tokens in phases:
        successes, failed_ids = run_batch(
            remaining, args, api_key, max_tokens, concurrency, label,
            results_by_id=results_by_id, save_fn=save, save_every=200,
        )
        save()

        attempted = {it["ID"] for it in remaining}
        now_good = {iid for iid in attempted
                    if iid in results_by_id and result_is_good(results_by_id[iid])}
        still_bad = (attempted - now_good) | set(failed_ids)
        remaining = [it for it in data if it["ID"] in still_bad]
        print(f"  Remaining after {label}: {len(remaining)}", flush=True)
        if not remaining:
            break

    if remaining:
        print(f"  {len(remaining)} items still have non-clean responses (kept in output).",
              flush=True)

    # Format summary
    counts: dict[str, int] = {}
    for it in results_by_id.values():
        k = it.get("format_classification", "unknown")
        counts[k] = counts.get(k, 0) + 1
    print(f"  Format summary: {counts}", flush=True)


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_input(raw: str):
    """Return (list_of_input_files, judge_output_dir)."""
    p = Path(raw).resolve()

    if p.is_file():
        if not p.name.endswith("_for_judging.json"):
            sys.exit(f"File must end with _for_judging.json: {p}")
        # for_judging/ parent → its parent is where judge/ goes
        judge_dir = p.parent.parent / "judge"
        return [str(p)], str(judge_dir)

    if p.is_dir():
        files = sorted(p.glob("*_for_judging.json"))
        if not files:
            sys.exit(f"No *_for_judging.json files found in {p}")
        judge_dir = p.parent / "judge"
        return [str(f) for f in files], str(judge_dir)

    sys.exit(f"Not a file or directory: {raw}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Judge DashBench _for_judging.json files via OpenRouter / OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="_for_judging.json file or folder containing them")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3-0324")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "openai"])
    parser.add_argument("--api-key-path", default="api_key/openrouter_key.txt",
                        help="text file that holds your own API key (OpenRouter key, or OpenAI key with --provider openai)")
    parser.add_argument("--strategy", default="yesno_system_only",
                        choices=["yesno_system_only", "yesno_system"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--low-concurrency", type=int, default=5)
    parser.add_argument("--bad-format-retry-wait", type=int, default=60)
    parser.add_argument("--retry-max-tokens", type=int, default=1024)
    parser.add_argument("--referer", default="https://localhost")
    parser.add_argument("--app-title", default="DashBench Judge")
    parser.add_argument("--judge-dir", default=None,
                        help="Override output directory (default: judge/ sibling of for_judging/)")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to every output filename before .json, e.g. '_newest'")
    parser.add_argument("--test", action="store_true",
                        help="Limit each file to 5 items for smoke-testing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be judged and the first judge request; no API call is made")
    args = parser.parse_args()

    input_files, judge_dir = resolve_input(args.input)
    if args.judge_dir:
        judge_dir = args.judge_dir
    Path(judge_dir).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for input_path in input_files:
            data = json.load(open(input_path))
            counts = {}
            for it in data:
                counts[it["prompt_type"]] = counts.get(it["prompt_type"], 0) + 1
            empty = [it["ID"] for it in data if it["prompt_type"] == "Q4" and it.get("extracted_results") is None]
            print(f"\nDRY RUN (no API call): {input_path}")
            print(f"  items: {len(data)}   per prompt type: {dict(sorted(counts.items()))}")
            print(f"  Q4 entries without parsable reasoning: {len(empty)}")
            print("  last lines of the first judge prompt:\n    " + data[0]["prompt"][-200:].replace("\n", "\n    "))
        print(f"\nWould write to: {judge_dir}")
        return

    if not os.path.exists(args.api_key_path):
        sys.exit(f"API key file not found: {args.api_key_path}\nCreate it with your own key, e.g.  mkdir -p api_key && "
                 f"echo YOUR_KEY > {args.api_key_path}   (or pass --api-key-path).")
    with open(args.api_key_path) as f:
        api_key = f.read().strip()

    print(f"Model    : {args.model}")
    print(f"Provider : {args.provider}")
    print(f"Strategy : {args.strategy}")
    print(f"Judge dir: {judge_dir}")
    print(f"Files    : {len(input_files)}")
    if args.test:
        print("TEST MODE: limiting each file to 5 items")

    for input_path in input_files:
        stem = Path(input_path).stem          # e.g. Qwen3_8B_for_judging
        output_path = os.path.join(judge_dir, f"judge_{stem}{args.output_suffix}.json")

        # In test mode, cap the data
        if args.test:
            with open(input_path) as f:
                orig = json.load(f)
            import tempfile, atexit
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(orig[:5], tmp)
            tmp.close()
            atexit.register(os.unlink, tmp.name)
            input_path = tmp.name

        judge_file(input_path, output_path, args, api_key)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
