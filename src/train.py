"""
train.py
Build a labeled training set from blocking candidates + ground truth,
train a classifier, and tune the decision threshold for F_0.5 (precision
weighted 2x over recall, per the challenge's evaluation formula).

Usage:
    python3 train.py --data-dir dataset/train --out-dir model \
        --sample-entities 0   # 0 = use all Source1 training entities

Design:
- Positive pairs: (S1 entity, true match) from train_ground_truth.tsv.
- Negative pairs: every OTHER candidate that blocking surfaced for that S1
  entity but is NOT a true match ("hard negatives" -- these are the pairs
  that actually look plausible, which is exactly what the model needs to
  learn to reject; random negatives would be too easy and teach nothing).
- Split by SOURCE1 ENTITY (not by pair) into train/val, so no entity's
  pairs leak across the split.
- Threshold tuned on validation to maximize macro-averaged F_0.5, matching
  the competition's actual scoring formula (including the empty-list-gets-
  1.0-for-a-singleton rule).
"""

import argparse
import csv
import pickle
import random
import sys
import time

sys.path.insert(0, "src") if "src" not in sys.path else None

from normalize import normalize_name, normalize_address, strip_entity_prefix
from blocking import BlockingIndex, truncate_candidates, MAX_CANDIDATES_PER_ENTITY
from store import RecordStore
from features import Featurizer, FEATURE_NAMES


def load_ground_truth(path):
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            matches = row["matched_entity_ids"]
            true_set = set()
            if matches:
                for mid in matches.split(","):
                    true_set.add((mid[:2], strip_entity_prefix(mid)))
            gt[row["source1_entity_id"]] = true_set
    return gt


def build_index_and_store(source2_path, source3_path, db_path):
    store = RecordStore(db_path, fresh=True)
    index = BlockingIndex()
    t0 = time.time()
    for src, path in [("S2", source2_path), ("S3", source3_path)]:
        n = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                iid = strip_entity_prefix(row["entity_id"])
                nn = normalize_name(row.get("business_name", ""))
                na = normalize_address(row.get("business_address", ""))
                country = row.get("country", "") or ""
                index.add((src, iid), nn, na)
                store.add(src, iid, nn, na, country)
                n += 1
                if n % 500000 == 0:
                    print(f"  ...{src}: {n} rows indexed ({time.time()-t0:.0f}s elapsed)")
        print(f"  {src}: {n} total rows")
    index.finalize()
    store.build_index()
    print(f"Index + store build: {time.time()-t0:.0f}s. Stats: {index.stats()}")
    return index, store


def macro_f_beta(tp, fp, fn, beta=0.5):
    """Per-entity F_beta, matching the challenge's formula exactly,
    including: an entity with no true matches scores 1.0 if predicted
    empty, 0.0 if predicted any match."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0  # correctly predicted empty for a true singleton
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) else 0.0


def evaluate_threshold(entity_predictions: dict, entity_true: dict, threshold: float) -> float:
    """entity_predictions: {s1_id: [(cand_key, score), ...]}
       entity_true: {s1_id: set of true cand_keys}
    Returns macro-averaged F_0.5 across all entities in entity_true."""
    scores = []
    for sid, true_set in entity_true.items():
        preds = entity_predictions.get(sid, [])
        pred_set = {cand for cand, score in preds if score >= threshold}
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        scores.append(macro_f_beta(tp, fp, fn))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--out-dir", default="model")
    ap.add_argument("--db-path", default="model/records.db")
    ap.add_argument("--sample-entities", type=int, default=0,
                     help="0 = use all Source1 training entities; N = random sample of N (for quick iteration)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-features", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading ground truth...")
    gt = load_ground_truth(f"{args.data_dir}/train_ground_truth.tsv")
    print(f"  {len(gt)} Source1 entities with ground truth")

    s1_ids = list(gt.keys())
    if args.sample_entities and args.sample_entities < len(s1_ids):
        s1_ids = random.sample(s1_ids, args.sample_entities)
        gt = {k: gt[k] for k in s1_ids}
        print(f"  sampled down to {len(s1_ids)} entities")

    print("Building blocking index + record store over Source2/3...")
    index, store = build_index_and_store(
        f"{args.data_dir}/train_source2.tsv",
        f"{args.data_dir}/train_source3.tsv",
        args.db_path,
    )

    print("Loading Source1 records for sampled entities...")
    s1_records = {}
    with open(f"{args.data_dir}/train_source1.tsv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        needed = set(s1_ids)
        for row in reader:
            if row["entity_id"] in needed:
                s1_records[row["entity_id"]] = row
    print(f"  loaded {len(s1_records)} Source1 records")

    print("Generating candidates for each sampled Source1 entity...")
    t0 = time.time()
    entity_candidates = {}  # sid -> list of (cand_key, cand_name, cand_addr, cand_country)
    all_needed_keys = set()
    entity_norm = {}
    for i, sid in enumerate(s1_ids):
        row = s1_records[sid]
        nn = normalize_name(row["business_name"])
        na = normalize_address(row["business_address"])
        entity_norm[sid] = (nn, na, row.get("country", ""))
        raw = index.candidates(nn, na)
        truncated = truncate_candidates(index, store, nn, na, raw, MAX_CANDIDATES_PER_ENTITY)
        entity_candidates[sid] = truncated
        all_needed_keys.update(truncated)
        if (i + 1) % 5000 == 0:
            print(f"  ...{i+1}/{len(s1_ids)} entities ({time.time()-t0:.0f}s)")
    print(f"Candidate generation: {time.time()-t0:.0f}s, {len(all_needed_keys)} unique candidate records needed")

    print("Fetching candidate records from store...")
    cand_records = store.get_batch(list(all_needed_keys))

    print("Building labeled pair list (positives + hard negatives)...")
    pairs = []          # (s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country)
    pair_meta = []       # (sid, cand_key, label)
    n_pos = n_neg = 0
    for sid in s1_ids:
        nn, na, nc = entity_norm[sid]
        true_set = gt[sid]
        for cand_key in entity_candidates[sid]:
            rec = cand_records.get(cand_key)
            if rec is None:
                continue
            cn, ca, cc = rec
            label = 1 if cand_key in true_set else 0
            pairs.append((nn, na, nc, cn, ca, cc))
            pair_meta.append((sid, cand_key, label))
            n_pos += label
            n_neg += (1 - label)
    print(f"  {len(pairs)} pairs total ({n_pos} positive, {n_neg} negative)")

    print("Fitting TF-IDF vectorizers (on a text sample, not the full pair corpus --")
    print("  fitting vocabulary/IDF only needs a representative sample; feeding every")
    print("  document in is slower and memory-heavier for no accuracy benefit)...")
    FIT_SAMPLE_SIZE = 120000
    all_names = list({p[0] for p in pairs} | {p[3] for p in pairs})
    all_addrs = list({p[1] for p in pairs} | {p[4] for p in pairs})
    random.shuffle(all_names)
    random.shuffle(all_addrs)
    name_corpus = all_names[:FIT_SAMPLE_SIZE]
    addr_corpus = all_addrs[:FIT_SAMPLE_SIZE]
    featurizer = Featurizer.fit(name_corpus, addr_corpus, max_features=args.max_features)

    print("Computing features (batched)...")
    t0 = time.time()
    feat_dicts = featurizer.compute_batch(pairs)
    print(f"  {time.time()-t0:.0f}s")

    import numpy as np
    X = np.array([[fd[name] for name in FEATURE_NAMES] for fd in feat_dicts])
    y = np.array([m[2] for m in pair_meta])
    sids = np.array([m[0] for m in pair_meta])
    cand_keys = [m[1] for m in pair_meta]

    # split by ENTITY, not by pair, to avoid leakage
    unique_sids = list(set(sids))
    random.shuffle(unique_sids)
    n_val = max(1, int(len(unique_sids) * args.val_frac))
    val_sids = set(unique_sids[:n_val])
    train_mask = np.array([s not in val_sids for s in sids])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train pairs: {len(X_train)} ({y_train.sum()} pos), Val pairs: {len(X_val)} ({y_val.sum()} pos)")

    print("Training classifier...")
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    class_weight="balanced", verbose=-1)
        model.fit(X_train, y_train)
        model_kind = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05,
                                                class_weight="balanced")
        model.fit(X_train, y_train)
        model_kind = "sklearn_histgb (lightgbm not installed -- pip install lightgbm for the real run)"
    print(f"  trained ({model_kind})")

    print("Scoring validation candidates and tuning threshold for F_0.5...")
    val_scores = model.predict_proba(X_val)[:, 1]
    val_sids_arr = sids[val_mask]
    val_cand_keys = [cand_keys[i] for i in range(len(cand_keys)) if val_mask[i]]

    entity_predictions = {}
    for sid, ckey, score in zip(val_sids_arr, val_cand_keys, val_scores):
        entity_predictions.setdefault(sid, []).append((ckey, score))
    entity_true_val = {sid: gt[sid] for sid in val_sids}

    best_thr, best_f05 = 0.5, -1
    for thr in [i / 100 for i in range(5, 96, 2)]:
        f05 = evaluate_threshold(entity_predictions, entity_true_val, thr)
        if f05 > best_f05:
            best_f05, best_thr = f05, thr
    print(f"Best threshold: {best_thr:.2f} -> local F_0.5 = {best_f05:.4f}")

    print("Saving model artifacts...")
    with open(f"{args.out_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    featurizer.save(f"{args.out_dir}/featurizer.pkl")
    with open(f"{args.out_dir}/threshold.txt", "w") as f:
        f.write(str(best_thr))
    print(f"Done. Model, featurizer, and threshold saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
