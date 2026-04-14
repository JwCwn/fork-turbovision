"""Inspect collected challenge URLs and metadata.

Usage:
    python scripts/collect_inspect.py <collected_dir>

Shows per-challenge URLs, prediction counts, action distribution, and any
duplicates. Output lists all URLs one per line to stdout (pipe to a file
for bulk download later).
"""

import json
import sys
from collections import Counter
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_inspect.py <collected_dir>")
        sys.exit(1)

    collect_dir = Path(sys.argv[1])
    jsonl = collect_dir / "challenges.jsonl"
    if not jsonl.exists():
        print(f"No challenges.jsonl at {jsonl}")
        sys.exit(1)

    entries = []
    with open(jsonl) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"=== {len(entries)} challenges recorded ===")
    if not entries:
        return

    # Deduplicate URLs
    urls = [e["video_url"] for e in entries]
    unique_urls = list(dict.fromkeys(urls))
    print(f"Unique URLs: {len(unique_urls)} (total recordings: {len(urls)})")

    # Action distribution across all predictions
    action_counter = Counter()
    pred_counts = []
    time_stats = []
    for e in entries:
        preds = e.get("predictions", [])
        pred_counts.append(len(preds))
        time_stats.append(e.get("processing_time_s", 0))
        for p in preds:
            action_counter[p["action"]] += 1

    print(f"\nPredictions per challenge: "
          f"avg={sum(pred_counts)/len(pred_counts):.1f} "
          f"min={min(pred_counts)} max={max(pred_counts)}")
    print(f"Processing time: "
          f"avg={sum(time_stats)/len(time_stats):.1f}s "
          f"min={min(time_stats):.1f}s max={max(time_stats):.1f}s")

    print("\n=== Total action counts across collected challenges ===")
    for action, count in action_counter.most_common():
        print(f"  {action}: {count}")

    # Write URLs for later bulk download
    urls_out = collect_dir / "urls.txt"
    with open(urls_out, "w") as f:
        for u in unique_urls:
            f.write(u + "\n")
    print(f"\nUnique URLs written to: {urls_out}")


if __name__ == "__main__":
    main()
