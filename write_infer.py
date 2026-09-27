"""Helper: writes the new src/infer.py v4."""
import os, textwrap

CODE = textwrap.dedent('''\
    """
    infer.py  v4  --  Multiprocessing + SQLite optimisations
    =========================================================
    Profiler results showed the bottleneck is SQLite I/O:
      1c_sql_query_blocking_db  73s of 184s total (40 pct)
      2_record_fetch_store_db   78s of 184s total (43 pct)
      5_feature_compute         28s of 184s total (15 pct)

    Fix: spawn N worker processes, each with its own SQLite connections.
    Workers also use mmap + 64MB cache PRAGMAs for faster repeated reads.
    Expected: 20-40 entities/sec on 12-core machine (was 2.7/sec).
    """
    import argparse
    import csv
    import os
    import pickle
    import sys
    import time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    import numpy as np

    from normalize import normalize_name, normalize_address
    from blocking import MAX_CANDIDATES_PER_ENTITY, name_tokens, all_keys
    from features import FEATURE_NAMES

    SQLITE_CHUNK = 900

    # Per-worker globals set once by _worker_init()
    _g_model = None
    _g_featurizer = None
    _g_threshold = None
    _g_index_conn = None   # blocking_index.db connection
    _g_store_conn = None   # records.db connection


    def _worker_init(model_ckpt_dir, test_ckpt_dir, threshold_override):
        """Called once per worker. Load model and open private SQLite connections."""
        global _g_model, _g_featurizer, _g_threshold, _g_index_conn, _g_store_conn
        import sqlite3
        from features import Featurizer

        model_dir = os.path.join(model_ckpt_dir, "model")
        with open(os.path.join(model_dir, "model.pkl"), "rb") as fh:
            _g_model = pickle.load(fh)
        _g_featurizer = Featurizer.load(os.path.join(model_ckpt_dir, "featurizer.pkl"))
        if threshold_override is not None:
            _g_threshold = threshold_override
        else:
            with open(os.path.join(model_dir, "threshold.txt")) as fh:
                _g_threshold = float(fh.read().strip())

        PRAGMAS = [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous=OFF;",
            "PRAGMA mmap_size=2147483648;",   # 2 GB memory-map
            "PRAGMA cache_size=-65536;",       # 64 MB page cache
            "PRAGMA temp_store=memory;",
        ]
        _g_index_conn = sqlite3.connect(
            os.path.join(test_ckpt_dir, "blocking_index.db"), check_same_thread=False)
        for p in PRAGMAS:
            _g_index_conn.execute(p)

        _g_store_conn = sqlite3.connect(
            os.path.join(test_ckpt_dir, "records.db"), check_same_thread=False)
        for p in PRAGMAS:
            _g_store_conn.execute(p)


    def _fmt_key(key):
        return f"{key[0]}-{key[1]}"


    def _quick_score(s1_name_tokens, s1_addr_tokens, cand_name, cand_address):
        """Same 60/40 weighted Jaccard as blocking.py."""
        cand_name_tok = name_tokens(cand_name)
        cand_addr_tok = frozenset(t for t in cand_address.split(" ") if t)
        n_union = s1_name_tokens | cand_name_tok
        a_union = s1_addr_tokens | cand_addr_tok
        n_score = len(s1_name_tokens & cand_name_tok) / len(n_union) if n_union else 0.0
        a_score = len(s1_addr_tokens & cand_addr_tok) / len(a_union) if a_union else 0.0
        return 0.6 * n_score + 0.4 * a_score


    def _batch_raw_candidates(keys_by_entity):
        """Batch SQLite query: one query per 900 keys instead of one per entity."""
        global _g_index_conn
        all_unique_keys = list({k for keys in keys_by_entity.values() for k in keys})
        if not all_unique_keys:
            return {eid: set() for eid in keys_by_entity}
        key_to_ids = {}
        for i in range(0, len(all_unique_keys), SQLITE_CHUNK):
            chunk = all_unique_keys[i: i + SQLITE_CHUNK]
            ph = ",".join("?" * len(chunk))
            q = (
                "SELECT block_key, source, entity_id "
                f"FROM block_entries WHERE block_key IN ({ph})"
            )
            for row in _g_index_conn.execute(q, chunk):
                key_to_ids.setdefault(row[0], []).append((row[1], row[2]))
        raw_by_entity = {}
        for orig_id, keys in keys_by_entity.items():
            found = set()
            for k in keys:
                lst = key_to_ids.get(k)
                if lst:
                    found.update(lst)
            raw_by_entity[orig_id] = found
        return raw_by_entity


    def _store_get_batch(key_ids):
        """Fetch records from records.db using this worker's private connection."""
        global _g_store_conn
        if not key_ids:
            return {}
        result = {}
        by_source = {}
        for src, eid in key_ids:
            by_source.setdefault(src, []).append(eid)
        for src, eids in by_source.items():
            for i in range(0, len(eids), SQLITE_CHUNK):
                chunk = eids[i: i + SQLITE_CHUNK]
                ph = ",".join("?" * len(chunk))
                q = (
                    "SELECT entity_id, name, address, country "
                    f"FROM records WHERE source=? AND entity_id IN ({ph})"
                )
                for row in _g_store_conn.execute(q, (src, *chunk)):
                    result[(src, row[0])] = (row[1], row[2], row[3])
        return result


    def _compute_features_numpy(featurizer, pairs):
        """Fill float32 matrix directly -- avoids building N Python dicts."""
        import difflib
        try:
            from rapidfuzz import fuzz as _rf
            _have_rf = True
        except ImportError:
            _have_rf = False
        from normalize import name_tokens as _name_tok, address_tokens as _addr_tok

        def _ratio(a, b):
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            if _have_rf:
                return _rf.ratio(a, b) / 100.0
            return difflib.SequenceMatcher(None, a, b).ratio()

        def _tsr(a, b):
            if _have_rf:
                return _rf.token_sort_ratio(a, b) / 100.0
            return _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

        def _tset(a, b):
            if _have_rf:
                return _rf.token_set_ratio(a, b) / 100.0
            ta, tb = set(a.split()), set(b.split())
            common = ta & tb
            base = " ".join(sorted(common))
            sa = (base + " " + " ".join(sorted(ta - tb))).strip()
            sb = (base + " " + " ".join(sorted(tb - ta))).strip()
            return max(_ratio(base, sa), _ratio(base, sb), _ratio(sa, sb))

        def _jacc(sa, sb):
            if not sa and not sb:
                return 1.0
            u = sa | sb
            return len(sa & sb) / len(u) if u else 0.0

        def _compact(a, b):
            ca, cb = a.replace(" ", ""), b.replace(" ", "")
            if not ca or not cb:
                return 0.0
            if ca == cb:
                return 1.0
            shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
            return len(shorter) / len(longer) if shorter in longer else _ratio(ca, cb)

        def _num_tok(text):
            return frozenset(t for t in text.split() if t.isdigit())

        n = len(pairs)
        X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i, (s1n, s1a, s1c, cn, ca, cc) in enumerate(pairs):
            s1nt = _name_tok(s1n)
            cnt  = _name_tok(cn)
            s1at = _addr_tok(s1a)
            cat  = _addr_tok(ca)
            la, lb = len(s1n), len(cn)
            ldr = 1.0 - abs(la - lb) / max(la, lb) if max(la, lb) else 1.0
            s1c_n = (s1c or "").strip().lower()
            cc_n  = (cc  or "").strip().lower()
            X[i, 0]  = _ratio(s1n, cn)
            X[i, 1]  = _tsr(s1n, cn)
            X[i, 2]  = _tset(s1n, cn)
            X[i, 3]  = _jacc(s1nt, cnt)
            X[i, 4]  = ldr
            X[i, 5]  = _compact(s1n, cn)
            # col 6: name_tfidf_cosine -- filled below
            X[i, 7]  = _ratio(s1a, ca)
            X[i, 8]  = _jacc(s1at, cat)
            X[i, 9]  = _jacc(_num_tok(s1a), _num_tok(ca))
            # col 10: addr_tfidf_cosine -- filled below
            X[i, 11] = float(bool(s1c_n) and s1c_n == cc_n)
            X[i, 12] = float(bool(s1c_n) and bool(cc_n))
            X[i, 13] = float(bool(s1n) and bool(cn) and s1n[0] == cn[0])

        # Batched TF-IDF: ONE transform call per vectorizer per side
        s1_names   = [p[0] for p in pairs]
        cand_names = [p[3] for p in pairs]
        s1_addrs   = [p[1] for p in pairs]
        cand_addrs = [p[4] for p in pairs]
        X[:, 6]  = featurizer._batch_cosine(featurizer.name_vectorizer, s1_names, cand_names)
        X[:, 10] = featurizer._batch_cosine(featurizer.addr_vectorizer, s1_addrs, cand_addrs)
        return X


    def _process_batch(batch_rows):
        """Process one batch: blocking -> fetch -> rank -> features -> predict."""
        global _g_model, _g_featurizer, _g_threshold

        # 1. Blocking keys (pure Python)
        keys_by_entity = {orig_id: all_keys(nn, na) for orig_id, nn, na, _ in batch_rows}

        # 2. Batch SQLite query for raw candidates
        raw_by_entity = _batch_raw_candidates(keys_by_entity)
        all_raw_keys = set()
        for raw in raw_by_entity.values():
            all_raw_keys.update(raw)

        # 3. Fetch all candidate records ONCE
        all_records = _store_get_batch(list(all_raw_keys))

        # 4. Truncate/rank using already-fetched records
        entity_candidates = {}
        for orig_id, nn, na, _ in batch_rows:
            raw = raw_by_entity[orig_id]
            if len(raw) <= MAX_CANDIDATES_PER_ENTITY:
                entity_candidates[orig_id] = list(raw)
                continue
            s1_name_tok = name_tokens(nn)
            s1_addr_tok = frozenset(t for t in na.split(" ") if t)
            scored = []
            for ck in raw:
                rec = all_records.get(ck)
                if rec is None:
                    continue
                scored.append((_quick_score(s1_name_tok, s1_addr_tok, rec[0], rec[1]), ck))
            scored.sort(reverse=True, key=lambda x: x[0])
            entity_candidates[orig_id] = [ck for _, ck in scored[:MAX_CANDIDATES_PER_ENTITY]]

        # 5. Build pair inputs from same records dict (no extra DB calls)
        pair_inputs = []
        pair_owner  = []
        for orig_id, nn, na, country in batch_rows:
            for ck in entity_candidates[orig_id]:
                rec = all_records.get(ck)
                if rec is None:
                    continue
                pair_inputs.append((nn, na, country, rec[0], rec[1], rec[2]))
                pair_owner.append((orig_id, ck))

        # 6. Feature matrix + model prediction
        if pair_inputs:
            X = _compute_features_numpy(_g_featurizer, pair_inputs)
            raw_scores = _g_model.predict_proba(X)[:, 1]
            del X
        else:
            raw_scores = np.array([], dtype=np.float32)

        # 7. Collect results
        matches_by_entity = {}
        for i, (orig_id, ck) in enumerate(pair_owner):
            if raw_scores[i] >= _g_threshold:
                matches_by_entity.setdefault(orig_id, []).append(ck)

        results = []
        for orig_id, _, _, _ in batch_rows:
            cand_list = entity_candidates.get(orig_id, [])
            cands_str = ",".join(_fmt_key(c) for c in cand_list)
            matched = matches_by_entity.get(orig_id, [])
            seen = set()
            matched_unique = [m for m in matched if not (m in seen or seen.add(m))]
            matches_str = ",".join(_fmt_key(m) for m in matched_unique)
            results.append((orig_id, cands_str, matches_str))
        return results


    def _process_chunk(args):
        """Worker entry point: split chunk into batches and call _process_batch."""
        chunk_rows, batch_size = args
        out = []
        for i in range(0, len(chunk_rows), batch_size):
            out.extend(_process_batch(chunk_rows[i: i + batch_size]))
        return out


    def main():
        ap = argparse.ArgumentParser(description="Entity resolution inference v4")
        ap.add_argument("--test-dir",       default="dataset/test")
        ap.add_argument("--test-ckpt-dir",  default="checkpoints_test")
        ap.add_argument("--model-ckpt-dir", default="checkpoints")
        ap.add_argument("--source1-file",   default="test_source1.tsv")
        ap.add_argument("--out-dir",        default="output")
        ap.add_argument("--batch-entities", type=int, default=2000,
                        help="Entities per internal batch per worker (2000-5000 recommended).")
        ap.add_argument("--threshold",      type=float, default=None)
        ap.add_argument("--limit-entities", type=int,   default=0,
                        help="0 = all entities.  Small N for benchmarking.")
        ap.add_argument("--workers",        type=int,   default=0,
                        help="Worker processes. 0 = auto (cpu_count - 1).")
        args = ap.parse_args()

        os.makedirs(args.out_dir, exist_ok=True)
        n_workers = args.workers or max(1, mp.cpu_count() - 1)
        print(f"Inference v4 -- {n_workers} workers on {mp.cpu_count()} cores")

        model_dir = os.path.join(args.model_ckpt_dir, "model")
        threshold = args.threshold
        if threshold is None:
            with open(os.path.join(model_dir, "threshold.txt")) as fh:
                threshold = float(fh.read().strip())
        print(f"  threshold = {threshold}")

        print("Reading Source1 entities into RAM...")
        t_read = time.time()
        all_rows = []
        source1_path = os.path.join(args.test_dir, args.source1_file)
        with open(source1_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                nn      = normalize_name(row.get("business_name", ""))
                na      = normalize_address(row.get("business_address", ""))
                country = row.get("country", "") or ""
                all_rows.append((row["entity_id"], nn, na, country))
                if args.limit_entities and len(all_rows) >= args.limit_entities:
                    break
        print(f"  {len(all_rows):,} entities in {time.time()-t_read:.1f}s")
        if not all_rows:
            print("Nothing to process.")
            return

        chunk_size = max(args.batch_entities,
                         (len(all_rows) + n_workers - 1) // n_workers)
        chunks = [all_rows[i: i + chunk_size]
                  for i in range(0, len(all_rows), chunk_size)]
        actual_workers = min(n_workers, len(chunks))
        print(f"  {len(chunks)} chunks, ~{chunk_size}/chunk, {actual_workers} workers")

        match_path = os.path.join(args.out_dir, "matching_results.tsv")
        cand_path  = os.path.join(args.out_dir, "candidate_pairs.tsv")
        t0 = time.time()
        n_total = n_with_matches = n_singleton = 0

        with (open(match_path, "w", newline="", encoding="utf-8") as match_out,
              open(cand_path,  "w", newline="", encoding="utf-8") as cand_out):
            mw = csv.writer(match_out, delimiter="\t")
            cw = csv.writer(cand_out,  delimiter="\t")
            mw.writerow(["source1_entity_id", "matched_entity_ids"])
            cw.writerow(["source1_entity_id", "candidate_entity_ids"])

            print("Spawning workers...")
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                initializer=_worker_init,
                initargs=(args.model_ckpt_dir, args.test_ckpt_dir, threshold),
            ) as pool:
                future_to_chunk = {
                    pool.submit(_process_chunk, (chunk, args.batch_entities)): ci
                    for ci, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_chunk):
                    ci = future_to_chunk[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        print(f"  [ERROR] chunk {ci}: {exc}", file=sys.stderr)
                        raise

                    for orig_id, cands_str, matches_str in results:
                        cw.writerow([orig_id, cands_str])
                        mw.writerow([orig_id, matches_str])
                        n_total += 1
                        if matches_str:
                            n_with_matches += 1
                        else:
                            n_singleton += 1

                    elapsed = time.time() - t0
                    rate    = n_total / elapsed if elapsed else 0.0
                    pct     = 100.0 * n_total / len(all_rows)
                    eta_h   = (len(all_rows) - n_total) / rate / 3600 if rate else 0.0
                    print(
                        f"  chunk {ci+1:3d}/{len(chunks)} -- "
                        f"{n_total:>8,}/{len(all_rows):,}  ({pct:5.1f}%)  "
                        f"{rate:5.1f} ent/sec  ETA {eta_h:.1f}h"
                    )

        elapsed = time.time() - t0
        rate = n_total / elapsed if elapsed else 0.0
        print("=" * 60)
        print(f"Done. {n_total:,} entities in {elapsed:.0f}s ({rate:.1f} ent/sec).")
        print(f"  Matched  : {n_with_matches:,}")
        print(f"  Singleton: {n_singleton:,}")
        print(f"Outputs: {match_path}")
        print(f"         {cand_path}")


    if __name__ == "__main__":
        main()
''')

target = os.path.join(os.path.dirname(__file__), "src", "infer.py")
with open(target, "w", encoding="utf-8") as f:
    f.write(CODE)
print(f"Written {len(CODE.splitlines())} lines to {target}")
