"""Turn the Hugging Face dataset into the frame folders and input files that main_api.py reads.

  python code/prepare_dataset.py                 # both splits, into dataset/
  python code/prepare_dataset.py --split ccw     # only the corner-case-window split

Output (the layout main_api.py expects):
  dataset/full_data.json   1,709 clips x 5 prompts = 8,545 entries (16 frames sampled across the whole clip)
  dataset/ccw_data.json      823 clips x 5 prompts = 4,115 entries (16 frames sampled inside the corner-case window)
  dataset/<MMAU|RoadSocial>/sampled_frames_16/<clip>/images/*.jpg      (full)
  dataset/<MMAU|RoadSocial>/corner_case_window/<clip>/images/*.jpg    (ccw)
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO = "gabrieldavin23/cornercase-dashbench"
SPLIT_DIR = {"full": "sampled_frames_16", "ccw": "corner_case_window"}


class Source:
    """Reads files either from the Hub or from a local copy of the repository (--hf-dir)."""

    def __init__(self, repo, hf_dir):
        self.repo, self.hf_dir = repo, hf_dir

    def files(self, prefix):
        if self.hf_dir:
            return sorted(str(p.relative_to(self.hf_dir)) for p in Path(self.hf_dir).glob(prefix + "*"))
        from huggingface_hub import list_repo_files
        return sorted(f for f in list_repo_files(self.repo, repo_type="dataset") if f.startswith(prefix))

    def get(self, name):
        if self.hf_dir:
            return str(Path(self.hf_dir) / name)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, name, repo_type="dataset")


def prepare(src, split, out, limit):
    clips = pq.read_table(src.get(f"data/{split}-00000-of-00001.parquet")).to_pylist()
    prompts = pq.read_table(src.get("prompts/train-00000-of-00001.parquet")).to_pylist()
    if limit:
        clips = clips[:limit]
    wanted = {c["id"]: c for c in clips}
    written = {}
    for shard in src.files(f"frames/{split}-"):
        if len(written) == len(wanted):
            break
        pf = pq.ParquetFile(src.get(shard))
        for batch in pf.iter_batches(batch_size=16, columns=["id", "frames"]):
            for row in batch.to_pylist():
                if row["id"] not in wanted or row["id"] in written:
                    continue
                c = wanted[row["id"]]
                dataset = "MMAU" if c["source"] == "MM-AU" else "RoadSocial"
                rel = Path(out) / dataset / SPLIT_DIR[split] / c["source_id"] / "images"
                rel.mkdir(parents=True, exist_ok=True)
                names = []
                for fr in row["frames"]:
                    (rel / fr["path"]).write_bytes(fr["bytes"])
                    names.append(fr["path"])
                written[row["id"]] = (dataset, str(rel), names)
        print(f"  {split}: {len(written)}/{len(wanted)} clips written", flush=True)
    missing = set(wanted) - set(written)
    if missing:
        raise SystemExit(f"{len(missing)} clips have no frames in the repository, e.g. {sorted(missing)[0]}")

    entries = []
    for c in clips:
        dataset, folder, names = written[c["id"]]
        for p in prompts:
            suffix = "_ccw" if split == "ccw" else ""
            entries.append({
                "ID": f"{c['id']}{suffix}_n_frames_16_{p['prompt_type']}",
                "dataset": dataset,
                "source_id": c["source_id"],
                "image_folder": folder,
                "n_frames": 16,
                "image_filenames": names,
                "prompt": p["prompt"],
                "prompt_type": p["prompt_type"],
                "ground_truth": c["ground_truth"],
                "category": c["category"],
            })
    target = Path(out) / f"{split}_data.json"
    target.write_text(json.dumps(entries, indent=2))
    print(f"{split}: {len(clips)} clips, {len(entries)} entries -> {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["full", "ccw", "both"], default="both")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--hf-dir", default=None, help="use a local copy of the repository instead of downloading")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (quick test)")
    args = ap.parse_args()
    src = Source(args.repo, args.hf_dir)
    for split in (["full", "ccw"] if args.split == "both" else [args.split]):
        prepare(src, split, args.out, args.limit)


if __name__ == "__main__":
    main()
