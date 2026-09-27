"""
stage4_compute_features.py
For each pair chunk: compute the similarity feature matrix and save as a
compact .npz (X, y, sids, cand_keys). Only ONE chunk's pairs and features
are ever in memory at a time -- this is what keeps memory bounded
regardless of total dataset size, unlike computing features for all pairs
in one big batch (which is what crashed earlier).

Resumable: chunks with an existing .npz are skipped on rerun.

Run:
    python3 src/stage4_compute_features.py --ckpt-dir checkpoints
"""
import argparse
import glob
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from features import Featurizer, FEATURE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    featurizer_path = os.path.join(args.ckpt_dir, "featurizer.pkl")
    if not os.path.exists(featurizer_path):
        print(f"ERROR: {featurizer_path} not found. Run stage3_fit_featurizer.py first.")
        sys.exit(1)
    featurizer = Featurizer.load(featurizer_path)

    pairs_dir = os.path.join(args.ckpt_dir, "pairs_chunks")
    feats_dir = os.path.join(args.ckpt_dir, "feature_chunks")
    os.makedirs(feats_dir, exist_ok=True)

    chunk_files = sorted(glob.glob(os.path.join(pairs_dir, "chunk_*.pkl")))
    print(f"Found {len(chunk_files)} pair chunks.")

    n_done = n_skipped = 0
    for path in chunk_files:
        chunk_name = os.path.basename(path).replace(".pkl", "")
        out_path = os.path.join(feats_dir, chunk_name + ".npz")
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        t0 = time.time()
        with open(path, "rb") as f:
            chunk_pairs = pickle.load(f)  # (s1n,s1a,s1c,cn,ca,cc,label,sid,cand_key)
        if not chunk_pairs:
            np.savez(out_path, X=np.zeros((0, len(FEATURE_NAMES))), y=np.zeros(0),
                      sids=np.array([]), cand_keys=np.array([]))
            n_done += 1
            continue

        feat_inputs = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in chunk_pairs]
        feat_dicts = featurizer.compute_batch(feat_inputs)
        X = np.array([[fd[name] for name in FEATURE_NAMES] for fd in feat_dicts], dtype=np.float32)
        y = np.array([p[6] for p in chunk_pairs], dtype=np.int8)
        sids = np.array([p[7] for p in chunk_pairs])
        cand_keys = np.array([f"{p[8][0]}-{p[8][1]}" for p in chunk_pairs])

        np.savez_compressed(out_path, X=X, y=y, sids=sids, cand_keys=cand_keys)
        n_done += 1
        print(f"[{chunk_name}] {len(chunk_pairs)} pairs -> features in {time.time()-t0:.1f}s")

    print(f"Stage 4: {n_done} chunks processed this run, {n_skipped} already done (skipped).")
    if n_skipped + n_done < len(chunk_files):
        print("  (incomplete -- rerun this exact command to continue)")
    else:
        print("  All chunks complete.")


if __name__ == "__main__":
    main()
