"""Turn a results file from main_api.py into the judge prompts of the paper (Figure 7).

  python code/convert_to_judging_format.py results/full/gpt-5-2025-08-07/full_gpt-5-2025-08-07.json

Q1, Q2, Q3 and Q5: the whole response is judged. Q4: the JSON is parsed; only "reasoning" is judged (Q4R) and
"unusual_or_dangerous_event_detected" is kept as event_detected (Q4TF). If the JSON cannot be parsed, nothing is judged
for that Q4 entry and it counts as incorrect.
Output: <results folder>/for_judging/<name>_for_judging.json
"""
import argparse
import json
import re
from pathlib import Path

TEMPLATE = """Your task is to decide whether the model-generated description captures the same core unusual or dangerous road event as the ground truth.
Focus specifically on whether the model matches the core cause, effect, and timing of the danger. Descriptions should agree on what unusual thing happened, when it happened, and why it was dangerous.
If the model omits the suddenness of the action, a major agent involved, or misrepresents who caused the danger, mark it as a mismatch.
You may ignore small differences in phrasing or object types, but do not accept answers that are vague, incomplete, or shift the core focus of the event.
First analyze, then report the final answer in either of Yes or No followed by the detailed explanation.
Examples:
Yes (if the model captures the same unusual or dangerous event)
---detailed explanation---
No (if it misses or distorts the key causal or temporal element)
---detailed explanation---

Model response: {extracted_results}

Ground truth: {ground_truth}"""


def extract_json(text):
    """First JSON object in the response that parses."""
    if not text or not isinstance(text, str):
        return "", False
    for pat in (r"```json\n(.*?)\n```", r"```\n(\{.*?\})\n```", r"(\{.*?\})"):
        for m in re.findall(pat, text, re.DOTALL)[:1]:
            try:
                json.loads(m.strip())
                return m.strip(), True
            except json.JSONDecodeError:
                continue
    return text.strip(), False


def q4_fields(text):
    s, _ = extract_json(text)
    try:
        d = json.loads(s)
        return d.get("reasoning", None), d.get("unusual_or_dangerous_event_detected", None)
    except Exception:
        return None, None


def response_text(api_response):
    try:
        return api_response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def to_item(entry):
    text = response_text(entry.get("api_response"))
    if entry["prompt_type"] == "Q4":
        extracted, detected = q4_fields(text)
    else:
        extracted, detected = text, None
    return {"ID": entry["ID"], "prompt_type": entry["prompt_type"], "ground_truth": entry["ground_truth"],
            "category": entry["category"],
            "event_detected": None if detected is None else str(detected),
            "extracted_results": extracted, "response": text,
            "prompt": TEMPLATE.format(extracted_results=extracted, ground_truth=entry["ground_truth"])}


def convert(path, out=None):
    path = Path(path)
    items = [to_item(e) for e in json.load(open(path))]
    out = Path(out) if out else path.parent / "for_judging" / f"{path.stem}_for_judging.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(out, "w"), indent=2)
    print(f"{len(items)} items -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results JSON written by main_api.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    convert(a.results, a.out)
