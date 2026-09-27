"""
stage2_generate_pairs.py
For each Source1 training entity: generate blocking candidates, label them
against ground truth (positive = true match, negative = candidate but not
a true match), and write to CHUNKED checkpoint files.

Resumable by design: Source1 entities are split into fixed-size chunks in
a deterministic order (file order), and each chunk writes its own output
file + a ".done" marker. On rerun, any chunk whose marker already exists
is skipped -- so if the process gets killed/times out partway through,
just run the exact same command again and it picks up where it left off.

Run:
    python3 src/stage2_generate_pairs.py --data-dir dataset/train \
        --ckpt-dir checkpoints --chunk-size 2000 [--limit-entities N]
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


def load_ground_truth(path, limit=None):
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
            if limit and len(gt) >= limit:
                break
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--source1-file", default="train_source1.tsv")
    ap.add_argument("--gt-file", default="train_ground_truth.tsv")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit-entities", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    index_path = os.path.join(args.ckpt_dir, "index.pkl")
    db_path = os.path.join(args.ckpt_dir, "records.db")
    pairs_dir = os.path.join(args.ckpt_dir, "pairs_chunks")
    os.makedirs(pairs_dir, exist_ok=True)

    if not os.path.exists(index_path):
        print(f"ERROR: {index_path} not found. Run stage1_build_index.py first.")
        sys.exit(1)

    print("Loading index...")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    store = RecordStore(db_path, fresh=False)

    print("Loading ground truth...")
    gt = load_ground_truth(os.path.join(args.data_dir, args.gt_file),
                            limit=args.limit_entities or None)
    s1_ids = list(gt.keys())
    print(f"  {len(s1_ids)} Source1 entities")

    chunks = [s1_ids[i:i + args.chunk_size] for i in range(0, len(s1_ids), args.chunk_size)]
    print(f"  {len(chunks)} chunks of up to {args.chunk_size} entities each")

    print("Loading Source1 records (streamed, filtered to needed ids)...")
    needed = set(s1_ids)
    s1_records = {}
    with open(os.path.join(args.data_dir, args.source1_file), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in needed:
                s1_records[row["entity_id"]] = row
    print(f"  loaded {len(s1_records)} records")

    n_skipped = n_done = 0
    for chunk_idx, chunk_ids in enumerate(chunks):
        out_path = os.path.join(pairs_dir, f"chunk_{chunk_idx:05d}.pkl")
        done_marker = out_path + ".done"
        if os.path.exists(done_marker):
            n_skipped += 1
            continue

        t0 = time.time()
        chunk_pairs = []  # (s1n, s1a, s1c, cn, ca, cc, label, sid, cand_key)
        for sid in chunk_ids:
            row = s1_records.get(sid)
            if row is None:
                continue
            nn = normalize_name(row["business_name"])
            na = normalize_address(row["business_address"])
            ncountry = row.get("country", "")
            true_set = gt[sid]

            raw = index.candidates(nn, na)
            truncated = truncate_candidates(index, store, nn, na, raw, MAX_CANDIDATES_PER_ENTITY)
            cand_keys = list(set(truncated) | true_set)  # ensure true matches always included even if truncation dropped them
            if not cand_keys:
                continue
            records = store.get_batch(cand_keys)
            for cand_key in cand_keys:
                rec = records.get(cand_key)
                if rec is None:
                    continue
                cn, ca, cc = rec
                label = 1 if cand_key in true_set else 0
                chunk_pairs.append((nn, na, ncountry, cn, ca, cc, label, sid, cand_key))

        with open(out_path, "wb") as f:
            pickle.dump(chunk_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(done_marker, "w") as f:
            f.write(f"{len(chunk_pairs)} pairs in {time.time()-t0:.1f}s\n")

        n_done += 1
        print(f"[{chunk_idx+1}/{len(chunks)}] {len(chunk_pairs)} pairs, {time.time()-t0:.1f}s")

    print(f"Stage 2: {n_done} chunks processed this run, {n_skipped} already done (skipped).")
    if n_skipped + n_done < len(chunks):
        print("  (incomplete -- rerun this exact command to continue)")
    else:
        print("  All chunks complete.")


if __name__ == "__main__":
    main()
