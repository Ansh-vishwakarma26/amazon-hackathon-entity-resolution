"""
stage3_fit_featurizer.py
Fit TF-IDF vectorizers on a SAMPLE of text drawn from the pair chunks
(not every document -- see the note in train.py/features.py: fitting
vocabulary/IDF only needs a representative sample, and feeding millions
of documents in is slower and memory-heavier for no benefit).

Run:
    python3 src/stage3_fit_featurizer.py --ckpt-dir checkpoints
"""
import argparse
import glob
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from features import Featurizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--sample-size", type=int, default=150000)
    ap.add_argument("--max-features", type=int, default=30000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = os.path.join(args.ckpt_dir, "featurizer.pkl")
    done_marker = out_path + ".done"
    if os.path.exists(done_marker) and not args.force:
        print(f"[SKIP] Stage 3 already complete (found {done_marker}).")
        return

    pairs_dir = os.path.join(args.ckpt_dir, "pairs_chunks")
    chunk_files = sorted(glob.glob(os.path.join(pairs_dir, "chunk_*.pkl")))
    if not chunk_files:
        print(f"ERROR: no pair chunks found in {pairs_dir}. Run stage2 first.")
        sys.exit(1)

    print(f"Sampling text from {len(chunk_files)} chunks (target {args.sample_size} texts)...")
    names, addrs = set(), set()
    random.shuffle(chunk_files)
    for path in chunk_files:
        with open(path, "rb") as f:
            chunk_pairs = pickle.load(f)
        for p in chunk_pairs:
            names.add(p[0]); names.add(p[3])
            addrs.add(p[1]); addrs.add(p[4])
        if len(names) >= args.sample_size and len(addrs) >= args.sample_size:
            break

    names, addrs = list(names), list(addrs)
    random.shuffle(names); random.shuffle(addrs)
    names = names[:args.sample_size]
    addrs = addrs[:args.sample_size]
    print(f"  fitting on {len(names)} name texts, {len(addrs)} address texts")

    featurizer = Featurizer.fit(names, addrs, max_features=args.max_features)
    featurizer.save(out_path)
    with open(done_marker, "w") as f:
        f.write(f"name_vocab={len(featurizer.name_vectorizer.vocabulary_)} "
                f"addr_vocab={len(featurizer.addr_vectorizer.vocabulary_)}\n")
    print(f"Stage 3 done. Saved {out_path}")


if __name__ == "__main__":
    main()
