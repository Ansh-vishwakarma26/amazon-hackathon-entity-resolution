"""
eval_true_f05.py
The HONEST end-to-end metric: for a sample of Source1 training entities,
run the EXACT inference path (raw blocking -> truncate -> trained model ->
threshold) and score against ground truth. Unlike stage5's reported
validation F0.5, this does NOT force true matches into the candidate set
first -- so it correctly reflects any recall lost to blocking.

Run:
    python3 src/eval_true_f05.py --data-dir dataset/train --ckpt-dir checkpoints [--limit-entities N]
"""
import argparse
import csv
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from normalize import normalize_name, normalize_address, strip_entity_prefix
from blocking import truncate_candidates, MAX_CANDIDATES_PER_ENTITY
from store import RecordStore
from features import Featurizer, FEATURE_NAMES


def macro_f_beta(tp, fp, fn, beta=0.5):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--limit-entities", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=None, help="override; default reads checkpoints/model/threshold.txt")
    args = ap.parse_args()

    model_dir = os.path.join(args.ckpt_dir, "model")
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    threshold = args.threshold
    if threshold is None:
        with open(os.path.join(model_dir, "threshold.txt")) as f:
            threshold = float(f.read().strip())
    featurizer = Featurizer.load(os.path.join(args.ckpt_dir, "featurizer.pkl"))

    with open(os.path.join(args.ckpt_dir, "index.pkl"), "rb") as f:
        index = pickle.load(f)
    store = RecordStore(os.path.join(args.ckpt_dir, "records.db"), fresh=False)

    gt = {}
    with open(os.path.join(args.data_dir, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
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

    needed = set(gt.keys())
    s1_records = {}
    with open(os.path.join(args.data_dir, "train_source1.tsv"), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in needed:
                s1_records[row["entity_id"]] = row

    print(f"Evaluating {len(gt)} entities end-to-end (threshold={threshold})...")
    t0 = time.time()
    f_scores = []
    total_tp = total_fp = total_fn = 0
    n_processed = 0

    for sid, true_set in gt.items():
        row = s1_records.get(sid)
        if row is None:
            continue
        nn = normalize_name(row["business_name"])
        na = normalize_address(row["business_address"])
        ncountry = row.get("country", "")

        raw = index.candidates(nn, na)
        truncated = truncate_candidates(index, store, nn, na, raw, MAX_CANDIDATES_PER_ENTITY)

        if not truncated:
            pred_set = set()
        else:
            records = store.get_batch(truncated)
            pair_inputs, keys = [], []
            for ck in truncated:
                rec = records.get(ck)
                if rec is None:
                    continue
                cn, ca, cc = rec
                pair_inputs.append((nn, na, ncountry, cn, ca, cc))
                keys.append(ck)
            if pair_inputs:
                feat_dicts = featurizer.compute_batch(pair_inputs)
                import numpy as np
                X = np.array([[fd[name] for name in FEATURE_NAMES] for fd in feat_dicts])
                scores = model.predict_proba(X)[:, 1]
                pred_set = {keys[i] for i in range(len(keys)) if scores[i] >= threshold}
            else:
                pred_set = set()

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp; total_fp += fp; total_fn += fn
        f_scores.append(macro_f_beta(tp, fp, fn))

        n_processed += 1
        if n_processed % 1000 == 0:
            print(f"  ...{n_processed}/{len(gt)} ({time.time()-t0:.0f}s)")

    macro_f05 = sum(f_scores) / len(f_scores)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    print(f"\n=== HONEST END-TO-END EVALUATION ({n_processed} entities, {time.time()-t0:.0f}s) ===")
    print(f"Macro-averaged F_0.5 (the actual competition metric): {macro_f05:.4f}")
    print(f"Micro precision: {micro_precision:.4f}, micro recall: {micro_recall:.4f}")
    print(f"Total TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Threshold used: {threshold}")


if __name__ == "__main__":
    main()
