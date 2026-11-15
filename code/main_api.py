"""Run a model on DashBench.

Every model is reached through an OpenAI-compatible endpoint: the OpenAI API (GPT-5), OpenRouter (Gemini-3-Flash) or a
local server such as vLLM. Responses are saved to results/<task>/<model>/<task>_<model>.json, and re-running the same
command resumes any missing entries.

Examples (from the repository root, after code/prepare_dataset.py):
  python code/main_api.py --model gpt-5-2025-08-07 --split full
  python code/main_api.py --model google/gemini-3-flash-preview --split ccw \\
      --openai-base-url https://openrouter.ai/api/v1 --api-key-path api_key/openrouter_key.txt
  python code/main_api.py --model Qwen/Qwen3-VL-8B-Instruct --split full --input-mode video \\
      --openai-base-url http://localhost:8000/v1
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai_api  # noqa: E402

SYSTEM_PROMPT = ("You are a helpful driving-scene assistant. Please be brief and concise in your description. "
                 "Answer within 250 words.")
SPLIT_FILES = {"full": "dataset/full_data.json", "ccw": "dataset/ccw_data.json"}
QUESTION_TYPES = ("Q1", "Q2", "Q3", "Q4", "Q5")


def read_api_key(path, base_url):
    """Read the API key from a text file. A local server (e.g. vLLM) needs no key."""
    if os.path.exists(path):
        return open(path).read().strip()
    if base_url and any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        return "EMPTY"
    sys.exit(f"API key file not found: {path}\nCreate it with your own key, e.g.  mkdir -p api_key && "
             f"echo YOUR_KEY > {path}   (or pass --api-key-path).")


def image_paths_of(item):
    return [os.path.join(item["image_folder"], f) for f in item["image_filenames"]]


def frames_to_temp_video(paths, fps=1):
    """Stitch the ordered frames into a temporary 1 fps mp4 with ffmpeg (video input mode)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --input-mode video")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\nduration {1 / fps}\n")
        concat = f.name
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out.close()
    try:
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-vf", f"fps={fps}", out.name], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")
    finally:
        os.unlink(concat)
    return out.name


def request_kwargs(item, args):
    schema = None
    if args.guided_json:
        import guided_schemas
        schema = guided_schemas.PROMPT_TYPE_SCHEMAS.get(item["prompt_type"].upper())
    return dict(model=args.model, prompt=item["prompt"], image_paths=image_paths_of(item),
                temperature=args.temperature, guided_json=schema, system_prompt=args.system_prompt,
                system_in_user=args.system_in_user, disable_thinking=args.disable_thinking,
                max_tokens=args.max_tokens)


def process_item(item, api_key, args):
    video_path = None
    try:
        kw = request_kwargs(item, args)
        if args.input_mode == "video":
            video_path = frames_to_temp_video(kw["image_paths"])
        out = dict(item)
        out["api_response"] = openai_api.make_api_call(api_key, args.openai_base_url, video_path=video_path, **kw)
        print(f"ok   {item['ID']}")
        return out
    except Exception as e:
        print(f"FAIL {item['ID']}: {e}")
        return None
    finally:
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)


def dry_run(data, args):
    """Check the inputs and build requests for a few items. Nothing is sent and nothing is paid for."""
    print("\nDRY RUN: no API call is made.")
    missing = [p for it in data for p in image_paths_of(it) if not os.path.exists(p)]
    print(f"items: {len(data)}   frame files missing: {len(missing)}")
    if missing:
        print("  e.g.", missing[0])
    by_type = {}
    for it in data:
        by_type[it["prompt_type"]] = by_type.get(it["prompt_type"], 0) + 1
    print("items per prompt type:", dict(sorted(by_type.items())))
    if args.input_mode == "video":
        print("ffmpeg found:", shutil.which("ffmpeg") is not None)
    for it in data[:2]:
        params = openai_api.build_request(**request_kwargs(it, args))
        content = params["messages"][-1]["content"]
        kinds = [c["type"] for c in content]
        size = sum(len(json.dumps(m)) for m in params["messages"])
        print(f"\n{it['ID']}: roles={[m['role'] for m in params['messages']]}  parts={kinds.count('image_url')} image_url / "
              f"{kinds.count('text')} text  payload~{size / 1e6:.2f} MB  temperature={params.get('temperature', 'default')}")
        print("  prompt:", it["prompt"][:90].replace("\n", " "))
    print(f"\nWould write: {args.output_file}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id, e.g. gpt-5-2025-08-07 or Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--split", choices=SPLIT_FILES, default="full",
                    help="full = 16 frames across the whole clip (1,709 clips); ccw = 16 frames inside the corner-case window (823 clips)")
    ap.add_argument("--input-file", default=None, help="override the input JSON of the split")
    ap.add_argument("--openai-base-url", default=None,
                    help="endpoint for OpenRouter or a local server; leave empty for the OpenAI API")
    ap.add_argument("--api-key-path", default="api_key/openai_key.txt", help="text file that holds your API key")
    ap.add_argument("--task-name", default=None, help="results sub-folder (default: the split name)")
    ap.add_argument("--output-file", default=None)
    ap.add_argument("--question-types", default="all", help="all, or a comma-separated list such as Q1,Q3")
    ap.add_argument("--input-mode", choices=["auto", "images", "video"], default="auto",
                    help="images: one image_url per frame. video: the 16 frames as one 1 fps video_url (needs ffmpeg; "
                         "local vLLM servers only). auto: video for a local server, images for OpenAI and OpenRouter")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    ap.add_argument("--system-in-user", action="store_true", help="put the system prompt in the user turn (e.g. InternVL3)")
    ap.add_argument("--guided-json", action="store_true", help="vLLM guided decoding for Q4")
    ap.add_argument("--disable-thinking", action="store_true", help="turn off thinking mode on models that support it")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--test", action="store_true", help="only the first 30 entries")
    ap.add_argument("--dry-run", action="store_true", help="check inputs and build requests, but make no API call")
    args = ap.parse_args()

    if args.input_mode == "auto":
        local = args.openai_base_url and any(h in args.openai_base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        args.input_mode = "video" if local else "images"
    task = args.task_name or args.split
    args.output_file = args.output_file or \
        f"results/{task}/{openai_api.model_folder_name(args.model)}/{task}_{openai_api.model_folder_name(args.model)}.json"

    input_file = args.input_file or SPLIT_FILES[args.split]
    if not os.path.exists(input_file):
        sys.exit(f"{input_file} not found. Run  python code/prepare_dataset.py --split {args.split}  first.")
    data = json.load(open(input_file))
    if args.question_types.lower() != "all":
        wanted = {q.strip().upper() for q in args.question_types.split(",")}
        bad = wanted - set(QUESTION_TYPES)
        if bad:
            sys.exit(f"Unknown question types {sorted(bad)}; valid: {', '.join(QUESTION_TYPES)}")
        data = [d for d in data if d["prompt_type"].upper() in wanted]
    if args.test:
        data = data[:30]

    print(f"Model: {args.model}\nSplit: {args.split}   input mode: {args.input_mode}\nEndpoint: "
          f"{args.openai_base_url or 'OpenAI API'}\nInput: {input_file} ({len(data)} entries)")
    if args.dry_run:
        return dry_run(data, args)

    api_key = read_api_key(args.api_key_path, args.openai_base_url)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(args.output_file)) if os.path.exists(args.output_file) else []
    done = {r["ID"] for r in results}
    todo = [d for d in data if d["ID"] not in done]
    print(f"{len(results)} existing results, {len(todo)} to run (concurrency {args.concurrency})")

    lock = threading.Lock()
    def save():
        with lock:
            json.dump(results, open(args.output_file, "w"), indent=2)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_item, it, api_key, args) for it in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is None:
                failed += 1
            else:
                results.append(r)
            if n % 50 == 0:
                save()
    save()
    print(f"Saved {len(results)} results to {args.output_file} ({failed} failed; run the command again to retry them)")


if __name__ == "__main__":
    main()
