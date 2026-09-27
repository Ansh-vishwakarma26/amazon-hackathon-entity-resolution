"""
stage5_train_model.py

Load all feature chunks, train the classifier, tune the decision threshold
to maximize macro-averaged F_0.5 (the challenge's actual scoring formula),
and save model + threshold.

Run:

    python src/stage5_train_model.py --ckpt-dir checkpoints --data-dir ..\data
"""

import argparse
import glob
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from features import FEATURE_NAMES


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

    return (
        (1 + b2) * precision * recall / denom
        if denom
        else 0.0
    )


def evaluate_threshold(
    sids,
    cand_keys,
    scores,
    y_true_lookup,
    threshold
):
    from collections import defaultdict

    preds_by_entity = defaultdict(set)

    for sid, ck, score in zip(sids, cand_keys, scores):
        if score >= threshold:
            preds_by_entity[sid].add(ck)

    f_scores = []

    for sid, true_set in y_true_lookup.items():
        pred_set = preds_by_entity.get(sid, set())

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        f_scores.append(
            macro_f_beta(tp, fp, fn)
        )

    return (
        sum(f_scores) / len(f_scores)
        if f_scores
        else 0.0
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ckpt-dir",
        default="checkpoints"
    )

    ap.add_argument(
        "--data-dir",
        default="dataset/train"
    )

    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.15
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = ap.parse_args()

    random.seed(args.seed)

    feats_dir = os.path.join(
        args.ckpt_dir,
        "feature_chunks"
    )

    chunk_files = sorted(
        glob.glob(
            os.path.join(
                feats_dir,
                "chunk_*.npz"
            )
        )
    )

    if not chunk_files:
        print(
            f"ERROR: no feature chunks in {feats_dir}. "
            "Run stage4 first."
        )
        sys.exit(1)

    print(
        f"Loading {len(chunk_files)} feature chunks..."
    )

    X_parts = []
    y_parts = []
    sid_parts = []
    ck_parts = []

    for path in chunk_files:

        d = np.load(
            path,
            allow_pickle=True
        )

        if len(d["y"]) == 0:
            continue

        X_parts.append(d["X"])
        y_parts.append(d["y"])
        sid_parts.append(d["sids"])
        ck_parts.append(d["cand_keys"])

    # ---------------------------------------------------------
    # Concatenate each dataset and immediately free the
    # original chunk lists to reduce peak RAM usage.
    # ---------------------------------------------------------

    X = np.concatenate(X_parts)
    del X_parts

    y = np.concatenate(y_parts)
    del y_parts

    sids = np.concatenate(sid_parts)
    del sid_parts

    cand_keys = np.concatenate(ck_parts)
    del ck_parts

    print(
        f"  total pairs: {len(y)} "
        f"({int(y.sum())} positive)"
    )

    # ---------------------------------------------------------
    # Load ground truth.
    #
    # np.unique() avoids converting all ~27.5M S1 IDs into
    # Python strings just to determine the unique entities.
    # ---------------------------------------------------------

    import csv

    unique_sids_arr = np.unique(sids)

    needed_sids = set(
        unique_sids_arr.tolist()
    )

    gt_lookup = {}

    with open(
        os.path.join(
            args.data_dir,
            "train_ground_truth.tsv"
        ),
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if row["source1_entity_id"] in needed_sids:

                matches = row[
                    "matched_entity_ids"
                ]

                gt_lookup[
                    row["source1_entity_id"]
                ] = (
                    set(matches.split(","))
                    if matches
                    else set()
                )

    # Reuse the already-computed unique IDs.
    unique_sids = unique_sids_arr.tolist()

    random.shuffle(unique_sids)

    n_val = max(
        1,
        int(
            len(unique_sids)
            * args.val_frac
        )
    )

    val_sids = set(
        unique_sids[:n_val]
    )

    # ---------------------------------------------------------
    # Create train/validation split.
    # ---------------------------------------------------------

    train_mask = np.array(
        [
            s not in val_sids
            for s in sids
        ]
    )

    val_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    sids_val = sids[val_mask]
    ck_val = cand_keys[val_mask]

    # ---------------------------------------------------------
    # The original full arrays are no longer needed.
    # Free them BEFORE model training.
    # ---------------------------------------------------------

    del X
    del y
    del sids
    del cand_keys
    del train_mask
    del val_mask

    print(
        f"Train: {len(X_train)} pairs, "
        f"Val: {len(X_val)} pairs"
    )

    print(
        "Training classifier..."
    )

    # ---------------------------------------------------------
    # Train LightGBM.
    # Fall back to HistGradientBoosting if LightGBM isn't
    # installed.
    # ---------------------------------------------------------

    try:

        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            verbose=-1
        )

        model.fit(
            X_train,
            y_train
        )

        model_kind = "lightgbm"

    except ImportError:

        from sklearn.ensemble import (
            HistGradientBoostingClassifier
        )

        model = HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced"
        )

        model.fit(
            X_train,
            y_train
        )

        model_kind = (
            "sklearn_histgb "
            "(install lightgbm for better/faster results)"
        )

    print(
        f"  trained ({model_kind})"
    )

    # ---------------------------------------------------------
    # Tune threshold for macro F_0.5.
    # ---------------------------------------------------------

    print(
        "Tuning threshold on validation set "
        "for macro F_0.5..."
    )

    val_scores = model.predict_proba(
        X_val
    )[:, 1]

    val_true_lookup = {
        sid: gt_lookup.get(
            sid,
            set()
        )
        for sid in val_sids
    }

    best_thr = 0.5
    best_f05 = -1

    for thr in [
        i / 100
        for i in range(5, 96, 2)
    ]:

        f05 = evaluate_threshold(
            sids_val,
            ck_val,
            val_scores,
            val_true_lookup,
            thr
        )

        if f05 > best_f05:

            best_f05 = f05
            best_thr = thr

    print(
        f"Best threshold: {best_thr:.2f} "
        f"-> local validation F_0.5 = {best_f05:.4f}"
    )

    # ---------------------------------------------------------
    # Save model, threshold and feature names.
    # ---------------------------------------------------------

    model_dir = os.path.join(
        args.ckpt_dir,
        "model"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    with open(
        os.path.join(
            model_dir,
            "model.pkl"
        ),
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )

    with open(
        os.path.join(
            model_dir,
            "threshold.txt"
        ),
        "w"
    ) as f:

        f.write(
            str(best_thr)
        )

    with open(
        os.path.join(
            model_dir,
            "feature_names.txt"
        ),
        "w"
    ) as f:

        f.write(
            "\n".join(
                FEATURE_NAMES
            )
        )

    print(
        f"Stage 5 done. "
        f"Model saved to {model_dir}/"
    )


if __name__ == "__main__":
    main()