"""
profile_infer.py -- TEMPORARY, READ-ONLY diagnostic tool.

Does NOT modify infer.py. Does NOT modify, rebuild, or finalize
checkpoints_test/*.db (opens with fresh=False, never calls .finalize()).
Does NOT write real output files (candidate/match strings are built, to
capture their real cost, then discarded -- nothing touches output/).

Uses the EXACT SAME logic as the current v2 infer.py (same functions,
same formulas, same cap of 150, same threshold handling) but wraps every
stage in time.perf_counter() so we measure where wall time actually goes,
instead of guessing. Reuses your real trained model/featurizer/threshold
and your real checkpoints_test index/store -- this is the real pipeline,
just instrumented.

Run (does not touch output/, safe to run anytime):
    python3 src/profile_infer.py --test-dir dataset/test \\
        --test-ckpt-dir checkpoints_test --model-ckpt-dir checkpoints \\
        --limit-entities 3000
"""

import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from normalize import normalize_name, normalize_address
from blocking import MAX_CANDIDATES_PER_ENTITY, DiskBlockingIndex, name_tokens, all_keys
from store import RecordStore
from features import Featurizer, FEATURE_NAMES

SQLITE_CHUNK = 900


def fmt_key(key):
    return f"{key[0]}-{key[1]}"


def quick_score(s1_name_tokens, s1_addr_tokens, cand_name, cand_address):
    """Byte-identical to the formula in blocking.py / infer.py v2."""
    cand_name_tokens = name_tokens(cand_name)
    cand_addr_tokens = frozenset(t for t in cand_address.split(" ") if t)
    n_union = s1_name_tokens | cand_name_tokens
    a_union = s1_addr_tokens | cand_addr_tokens
    n_score = len(s1_name_tokens & cand_name_tokens) / len(n_union) if n_union else 0
    a_score = len(s1_addr_tokens & cand_addr_tokens) / len(a_union) if a_union else 0
    return 0.6 * n_score + 0.4 * a_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--test-ckpt-dir", default="checkpoints_test")
    ap.add_argument("--model-ckpt-dir", default="checkpoints")
    ap.add_argument("--source1-file", default="test_source1.tsv")
    ap.add_argument("--batch-entities", type=int, default=500)
    ap.add_argument("--limit-entities", type=int, default=3000,
                     help="Sample size to profile. Keep this small (1000-5000) -- "
                          "this is a diagnostic run, not the real inference.")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    print("Loading model, featurizer, threshold (read-only)...")
    model_dir = os.path.join(args.model_ckpt_dir, "model")
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    threshold = args.threshold
    if threshold is None:
        with open(os.path.join(model_dir, "threshold.txt")) as f:
            threshold = float(f.read().strip())
    featurizer = Featurizer.load(os.path.join(args.model_ckpt_dir, "featurizer.pkl"))
    print(f"  threshold = {threshold}")

    print("Opening TEST index + store READ-ONLY (fresh=False, no finalize/rebuild call)...")
    index = DiskBlockingIndex(os.path.join(args.test_ckpt_dir, "blocking_index.db"), fresh=False)
    index._finalized = True  # trust Stage 1's existing finalize -- do NOT call finalize() here
    store = RecordStore(os.path.join(args.test_ckpt_dir, "records.db"), fresh=False)

    timers = defaultdict(float)
    counters = defaultdict(int)
    n_total = 0
    wall_t0 = time.perf_counter()

    def process_batch(batch_rows):
        nonlocal n_total

        # ---- Stage 1a: compute blocking keys locally (pure Python) ----
        t0 = time.perf_counter()
        keys_by_entity = {}
        for orig_id, nn, na, country in batch_rows:
            keys_by_entity[orig_id] = all_keys(nn, na)
        timers["1a_compute_keys"] += time.perf_counter() - t0

        # ---- Stage 1b: union keys across the batch ----
        t0 = time.perf_counter()
        all_unique_keys = set()
        for keys in keys_by_entity.values():
            all_unique_keys.update(keys)
        all_unique_keys_list = list(all_unique_keys)
        timers["1b_union_keys"] += time.perf_counter() - t0
        counters["unique_keys_per_batch"] += len(all_unique_keys_list)

        # ---- Stage 1c: the actual SQLite query against blocking_index.db ----
        t0 = time.perf_counter()
        key_to_ids = {}
        for i in range(0, len(all_unique_keys_list), SQLITE_CHUNK):
            chunk = all_unique_keys_list[i:i + SQLITE_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            query = f"SELECT block_key, source, entity_id FROM block_entries WHERE block_key IN ({placeholders})"
            for row in index.conn.execute(query, chunk):
                key_to_ids.setdefault(row[0], []).append((row[1], row[2]))
        timers["1c_sql_query_blocking_db"] += time.perf_counter() - t0
        counters["rows_returned_from_blocking_query"] += sum(len(v) for v in key_to_ids.values())

        # ---- Stage 1d: redistribute results back to each entity ----
        t0 = time.perf_counter()
        raw_by_entity = {}
        for orig_id, keys in keys_by_entity.items():
            found = set()
            for key in keys:
                if key in key_to_ids:
                    found.update(key_to_ids[key])
            raw_by_entity[orig_id] = found
        timers["1d_redistribute"] += time.perf_counter() - t0

        all_raw_keys = set()
        for raw in raw_by_entity.values():
            all_raw_keys.update(raw)
        counters["unique_raw_candidates_per_batch"] += len(all_raw_keys)
        counters["total_raw_candidate_slots"] += sum(len(r) for r in raw_by_entity.values())

        # ---- Stage 2: fetch every raw candidate's record ONCE for the batch ----
        t0 = time.perf_counter()
        all_records = store.get_batch(list(all_raw_keys))
        timers["2_record_fetch_store_db"] += time.perf_counter() - t0

        # ---- Stage 3: truncate/rank locally in Python ----
        t0 = time.perf_counter()
        entity_candidates = {}
        n_entities_ranked = 0
        n_scored_pairs = 0
        for orig_id, nn, na, country in batch_rows:
            raw = raw_by_entity[orig_id]
            if len(raw) <= MAX_CANDIDATES_PER_ENTITY:
                entity_candidates[orig_id] = list(raw)
                continue
            n_entities_ranked += 1
            s1_name_tok = name_tokens(nn)
            s1_addr_tok = frozenset(t for t in na.split(" ") if t)
            scored = []
            for ck in raw:
                rec = all_records.get(ck)
                if rec is None:
                    continue
                cand_name, cand_address, _country = rec
                score = quick_score(s1_name_tok, s1_addr_tok, cand_name, cand_address)
                scored.append((score, ck))
                n_scored_pairs += 1
            scored.sort(reverse=True, key=lambda x: x[0])
            entity_candidates[orig_id] = [ck for _s, ck in scored[:MAX_CANDIDATES_PER_ENTITY]]
        timers["3_truncate_rank"] += time.perf_counter() - t0
        counters["entities_needing_ranking"] += n_entities_ranked
        counters["scored_pairs_in_ranking"] += n_scored_pairs

        # ---- Stage 4: build feature inputs (bookkeeping) ----
        t0 = time.perf_counter()
        pair_inputs, pair_owner = [], []
        for orig_id, nn, na, country in batch_rows:
            for ck in entity_candidates[orig_id]:
                rec = all_records.get(ck)
                if rec is None:
                    continue
                cn, ca, cc = rec
                pair_inputs.append((nn, na, country, cn, ca, cc))
                pair_owner.append((orig_id, ck))
        timers["4_build_pair_inputs"] += time.perf_counter() - t0
        counters["feature_pairs"] += len(pair_inputs)

        # ---- Stage 5: feature computation (features.py compute_batch) ----
        t0 = time.perf_counter()
        feat_dicts = []
        if pair_inputs:
            feat_dicts = featurizer.compute_batch(pair_inputs)
        timers["5_feature_compute"] += time.perf_counter() - t0

        # ---- Stage 6: model prediction ----
        t0 = time.perf_counter()
        scores = np.array([], dtype=np.float32)
        if pair_inputs:
            X = np.array([[fd[name] for name in FEATURE_NAMES] for fd in feat_dicts], dtype=np.float32)
            scores = model.predict_proba(X)[:, 1]
        timers["6_model_predict"] += time.perf_counter() - t0

        # ---- Stage 7: output-row construction (NOT written to real files --
        # result is discarded, only measuring the cost of building the strings) ----
        t0 = time.perf_counter()
        matches_by_entity = {}
        for i, (orig_id, ck) in enumerate(pair_owner):
            if scores[i] >= threshold:
                matches_by_entity.setdefault(orig_id, []).append(ck)
        for orig_id, nn, na, country in batch_rows:
            cand_list = entity_candidates.get(orig_id, [])
            _ = ",".join(fmt_key(c) for c in cand_list)
            matched = matches_by_entity.get(orig_id, [])
            seen = set()
            matched_unique = [m for m in matched if not (m in seen or seen.add(m))]
            _ = ",".join(fmt_key(m) for m in matched_unique)
            n_total += 1
        timers["7_output_row_building"] += time.perf_counter() - t0
        print(f"  Batch complete: {n_total}/{args.limit_entities} entities", flush=True)

    print(f"Profiling {args.limit_entities} entities in batches of {args.batch_entities} "
          f"(READ-ONLY -- no files written, no index changes)...")
    batch = []
    source1_path = os.path.join(args.test_dir, args.source1_file)
    with open(source1_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            nn = normalize_name(row.get("business_name", ""))
            na = normalize_address(row.get("business_address", ""))
            country = row.get("country", "") or ""
            batch.append((row["entity_id"], nn, na, country))

            if args.limit_entities and (n_total + len(batch)) >= args.limit_entities:
                process_batch(batch)
                batch = []
                break

            if len(batch) >= args.batch_entities:
                process_batch(batch)
                batch = []

    wall_elapsed = time.perf_counter() - wall_t0

    print("\n" + "=" * 74)
    print(f"PROFILING REPORT -- {n_total} entities, {wall_elapsed:.1f}s wall time, "
          f"{n_total/wall_elapsed:.2f} entities/sec")
    print("=" * 74)
    stage_order = [
        "1a_compute_keys", "1b_union_keys", "1c_sql_query_blocking_db", "1d_redistribute",
        "2_record_fetch_store_db", "3_truncate_rank", "4_build_pair_inputs",
        "5_feature_compute", "6_model_predict", "7_output_row_building",
    ]
    accounted = sum(timers[s] for s in stage_order)
    for s in stage_order:
        pct = 100 * timers[s] / wall_elapsed if wall_elapsed else 0
        print(f"  {s:28s} {timers[s]:8.2f}s  ({pct:5.1f}%)")
    unaccounted = wall_elapsed - accounted
    pct_un = 100 * unaccounted / wall_elapsed if wall_elapsed else 0
    print(f"  {'(unaccounted: I/O, GC, etc.)':28s} {unaccounted:8.2f}s  ({pct_un:5.1f}%)")

    print("\nVolume counters (context for interpreting the stage costs above):")
    for k in ["unique_keys_per_batch", "rows_returned_from_blocking_query",
              "unique_raw_candidates_per_batch", "total_raw_candidate_slots",
              "entities_needing_ranking", "scored_pairs_in_ranking", "feature_pairs"]:
        print(f"  {k}: {counters[k]}")

    index.close()
    store.close()


if __name__ == "__main__":
    main()