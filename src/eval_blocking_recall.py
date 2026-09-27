"""
eval_blocking_recall.py
Measures GENUINE blocking recall: for each Source1 training entity, calls
the exact same index.candidates() + truncate_candidates() path that
inference will use (no shortcuts, no forced inclusion of true matches),
then checks how many true matches actually survive.

This is deliberately SEPARATE from stage2_generate_pairs.py, which forces
true matches into the candidate set so the classifier has positive
examples to train on. That's correct for training, but it means stage5's
reported F0.5 does NOT reflect blocking misses. This script measures the
number stage5 can't see.

Run:
    python3 src/eval_blocking_recall.py --data-dir dataset/train --ckpt-dir checkpoints [--limit-entities N]
"""
import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from normalize import normalize_name, normalize_address, strip_entity_prefix
from blocking import truncate_candidates, MAX_CANDIDATES_PER_ENTITY
from store import RecordStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--source1-file", default="train_source1.tsv")
    ap.add_argument("--gt-file", default="train_ground_truth.tsv")
    ap.add_argument("--limit-entities", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    index_path = os.path.join(args.ckpt_dir, "index.pkl")
    db_path = os.path.join(args.ckpt_dir, "records.db")
    if not os.path.exists(index_path):
        print(f"ERROR: {index_path} not found. Run stage1_build_index.py first.")
        sys.exit(1)

    print("Loading index + store...")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    store = RecordStore(db_path, fresh=False)

    print("Loading ground truth...")
    gt = {}
    with open(os.path.join(args.data_dir, args.gt_file), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            matches = row["matched_entity_ids"]
            true_set = set()
            if matches:
                for mid in matches.split(","):
                    true_set.add((mid[:2], strip_entity_prefix(mid)))
            gt[row["source1_entity_id"]] = true_set
            if args.limit_entities and len(gt) >= args.limit_entities:
                break
    print(f"  {len(gt)} entities loaded")

    needed = set(gt.keys())
    s1_records = {}
    with open(os.path.join(args.data_dir, args.source1_file), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in needed:
                s1_records[row["entity_id"]] = row

    t0 = time.time()
    total_true = total_found_raw = total_found_truncated = 0
    total_true_s2 = total_found_s2 = 0
    total_true_s3 = total_found_s3 = 0
    raw_cand_counts, truncated_cand_counts = [], []
    entities_with_missed_match = 0
    entities_singleton = entities_singleton_correct = 0
    lost_examples = []

    n = 0
    for sid, true_set in gt.items():
        row = s1_records.get(sid)
        if row is None:
            continue
        nn = normalize_name(row["business_name"])
        na = normalize_address(row["business_address"])

        raw = index.candidates(nn, na)                 # union of all blocking keys, uncapped
        truncated = truncate_candidates(index, store, nn, na, raw, MAX_CANDIDATES_PER_ENTITY)
        truncated_set = set(truncated)

        raw_cand_counts.append(len(raw))
        truncated_cand_counts.append(len(truncated))

        if not true_set:
            entities_singleton += 1
            if not truncated_set:
                entities_singleton_correct += 1
            n += 1
            continue

        found_raw = true_set & raw
        found_trunc = true_set & truncated_set
        total_true += len(true_set)
        total_found_raw += len(found_raw)
        total_found_truncated += len(found_trunc)

        for src, iid in true_set:
            if src == "S2":
                total_true_s2 += 1
                if (src, iid) in truncated_set:
                    total_found_s2 += 1
            else:
                total_true_s3 += 1
                if (src, iid) in truncated_set:
                    total_found_s3 += 1

        if len(found_trunc) < len(true_set):
            entities_with_missed_match += 1
            if len(lost_examples) < 10:
                missed = true_set - truncated_set
                lost_examples.append((sid, row["business_name"], missed))

        n += 1
        if n % 2000 == 0:
            print(f"  ...{n}/{len(gt)} ({time.time()-t0:.0f}s)")

    print(f"\n=== BLOCKING RECALL REPORT ({n} entities evaluated, {time.time()-t0:.0f}s) ===")
    print(f"Recall BEFORE truncation (raw candidates() union): {total_found_raw}/{total_true} = {total_found_raw/total_true:.4f}")
    print(f"Recall AFTER truncation to {MAX_CANDIDATES_PER_ENTITY}/entity (what inference actually uses): "
          f"{total_found_truncated}/{total_true} = {total_found_truncated/total_true:.4f}")
    print(f"  -- Source2 recall: {total_found_s2}/{total_true_s2} = {total_found_s2/total_true_s2 if total_true_s2 else float('nan'):.4f}")
    print(f"  -- Source3 recall: {total_found_s3}/{total_true_s3} = {total_found_s3/total_true_s3 if total_true_s3 else float('nan'):.4f}")
    print(f"Avg candidates/entity (raw): {sum(raw_cand_counts)/len(raw_cand_counts):.1f}, max: {max(raw_cand_counts)}")
    print(f"Avg candidates/entity (truncated): {sum(truncated_cand_counts)/len(truncated_cand_counts):.1f}, max: {max(truncated_cand_counts)}")
    print(f"Entities with >=1 true match LOST by blocking: {entities_with_missed_match}")
    print(f"Singleton entities (no true match): {entities_singleton}, correctly given 0 candidates: {entities_singleton_correct}")
    print(f"  (a singleton with >0 candidates isn't wrong by itself -- the MODEL still needs to reject")
    print(f"   all of them at inference time for that entity to score 1.0)")

    if lost_examples:
        print("\n--- Sample of entities where blocking lost a true match ---")
        for sid, name, missed in lost_examples:
            print(f"  {sid} ({name!r}): missed {missed}")


if __name__ == "__main__":
    main()
