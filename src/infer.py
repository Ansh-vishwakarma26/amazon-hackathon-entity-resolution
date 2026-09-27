import os
import sys
import time
import csv
import argparse
try:
    import psutil
except ImportError:
    psutil = None
import lightgbm as lgb

from normalize import normalize_name, normalize_address
from blocking import MAX_CANDIDATES_PER_ENTITY, DiskBlockingIndex, name_tokens, all_keys
from store import RecordStore
from features import Featurizer, FEATURE_NAMES

def quick_score(s1_name_tokens, s1_addr_tokens, cand_name, cand_address):
    cand_name_tokens = name_tokens(cand_name)
    cand_addr_tokens = frozenset(t for t in cand_address.split(' ') if t)
    
    name_intersect = len(s1_name_tokens & cand_name_tokens)
    name_union = len(s1_name_tokens | cand_name_tokens)
    jaccard_name = name_intersect / name_union if name_union > 0 else 0.0

    addr_intersect = len(s1_addr_tokens & cand_addr_tokens)
    addr_union = len(s1_addr_tokens | cand_addr_tokens)
    jaccard_addr = addr_intersect / addr_union if addr_union > 0 else 0.0

    return 0.7 * jaccard_name + 0.3 * jaccard_addr

def batch_raw_candidates(index, keys_by_entity: dict) -> dict:
    raw_by_entity = {e_id: set() for e_id in keys_by_entity}
    all_unique_keys = set()
    for keys in keys_by_entity.values():
        all_unique_keys.update(keys)
    
    if not all_unique_keys:
        return raw_by_entity

    key_to_cands = index.get_candidates_for_keys_batch(all_unique_keys)
    for e_id, keys in keys_by_entity.items():
        cands = set()
        for k in keys:
            if k in key_to_cands:
                cands.update(key_to_cands[k])
        raw_by_entity[e_id] = cands

    return raw_by_entity

def fmt_key(k):
    return str(k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dir', type=str, default='../data')
    ap.add_argument('--test-ckpt-dir', type=str, default='checkpoints')
    ap.add_argument('--out-dir', type=str, default='output')
    ap.add_argument('--model-path', type=str, default='checkpoints/model/model.pkl')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--batch-entities', type=int, default=1000)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--limit-entities', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.num_shards > 1:
        match_path = os.path.join(args.out_dir, f'matching_results_shard{args.shard_index}.tsv')
        cand_path = os.path.join(args.out_dir, f'candidate_pairs_shard{args.shard_index}.tsv')
    else:
        match_path = os.path.join(args.out_dir, 'matching_results.tsv')
        cand_path = os.path.join(args.out_dir, 'candidate_pairs.tsv')

    source1_path = os.path.join(args.test_dir, 'source1.tsv')
    if not os.path.exists(source1_path):
        source1_path = os.path.join(args.test_dir, 'test_source1.tsv')

    print(f'[shard {args.shard_index}] Opening TEST index + store...')
    index = DiskBlockingIndex(os.path.join(args.test_ckpt_dir, 'blocking_index.db'), fresh=False)
    index._finalized = True
    index.conn.execute('PRAGMA query_only = ON;')
    index.conn.execute('PRAGMA cache_size = -262144;')
    index.conn.execute('PRAGMA temp_store = MEMORY;')
    index.conn.execute('PRAGMA mmap_size = 30000000000;')

    store = RecordStore(os.path.join(args.test_ckpt_dir, 'records.db'), fresh=False)
    store.conn.execute('PRAGMA cache_size = -262144;')
    store.conn.execute('PRAGMA temp_store = MEMORY;')
    store.conn.execute('PRAGMA mmap_size = 30000000000;')

    if args.model_path.endswith('.pkl'):
        import joblib
        model = joblib.load(args.model_path)
    else:
        model = lgb.Booster(model_file=args.model_path)
    featurizer = Featurizer()

    process = psutil.Process(os.getpid()) if psutil else None
    peak_rss_mb = 0
    threshold = args.threshold if args.threshold is not None else 0.5

    def process_batch(batch_rows, match_writer, cand_writer):
        nonlocal peak_rss_mb
        if not batch_rows:
            return

        keys_by_entity = {}
        for orig_id, nn, na, country in batch_rows:
            keys_by_entity[orig_id] = all_keys(nn, na)

        raw_by_entity = batch_raw_candidates(index, keys_by_entity)

        all_raw_keys = set()
        for raw in raw_by_entity.values():
            all_raw_keys.update(raw)

        all_records = store.get_records_batch(all_raw_keys)
        cand_token_cache = {}
        entity_candidates = {}

        for orig_id, nn, na, country in batch_rows:
            raw = raw_by_entity[orig_id]
            if len(raw) <= MAX_CANDIDATES_PER_ENTITY:
                entity_candidates[orig_id] = list(raw)
                continue

            s1_name_tok = name_tokens(nn)
            s1_addr_tok = frozenset(t for t in na.split(' ') if t)
            scored = []
            for ck in raw:
                rec = all_records.get(ck)
                if rec:
                    score = quick_score(s1_name_tok, s1_addr_tok, rec[0], rec[1])
                    scored.append((score, ck))
            scored.sort(reverse=True, key=lambda x: x[0])
            entity_candidates[orig_id] = [ck for _s, ck in scored[:MAX_CANDIDATES_PER_ENTITY]]

        pair_inputs = []
        pair_owner = []
        for orig_id, nn, na, country in batch_rows:
            for ck in entity_candidates[orig_id]:
                rec = all_records.get(ck)
                if rec is None:
                    continue
                pair_inputs.append((nn, na, country, rec[0], rec[1], rec[2]))
                pair_owner.append((orig_id, ck))

        matched_by_entity = {orig_id: [] for orig_id, _, _, _ in batch_rows}

        if pair_inputs:
            X = featurizer.transform_pairs(pair_inputs)
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            for (orig_id, ck), prob in zip(pair_owner, preds):
                if prob >= threshold:
                    matched_by_entity[orig_id].append(ck)

        for orig_id, nn, na, country in batch_rows:
            matches = matched_by_entity.get(orig_id, [])
            match_writer.writerow([orig_id, ','.join(fmt_key(m) for m in matches)])

        for orig_id, nn, na, country in batch_rows:
            cand_list = entity_candidates.get(orig_id, [])
            cand_writer.writerow([orig_id, ','.join(fmt_key(c) for c in cand_list)])

        if process:
            rss_mb = process.memory_info().rss / (1024 * 1024)
            if rss_mb > peak_rss_mb:
                peak_rss_mb = rss_mb

        cand_token_cache.clear()

    t0 = time.time()
    with open(match_path, 'w', newline='', encoding='utf-8-sig') as match_out, open(cand_path, 'w', newline='', encoding='utf-8-sig') as cand_out:
        match_writer = csv.writer(match_out, delimiter='	')
        cand_writer = csv.writer(cand_out, delimiter='	')
        match_writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        cand_writer.writerow(['source1_entity_id', 'candidate_entity_ids'])

        n_seen = 0
        batch = []

        with open(source1_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='	')
            row_idx = 0
            for row in reader:
                this_row_idx = row_idx
                row_idx += 1

                if args.num_shards > 1 and (this_row_idx % args.num_shards) != args.shard_index:
                    continue

                n_seen += 1
                orig_id = row.get('entity_id') or row.get('id')
                raw_name = row.get('business_name') or row.get('name') or ''
                raw_addr = row.get('business_address') or row.get('address') or ''
                country = row.get('country', '')

                nn = normalize_name(raw_name)
                na = normalize_address(raw_addr)

                batch.append((orig_id, nn, na, country))

                if len(batch) >= args.batch_entities:
                    process_batch(batch, match_writer, cand_writer)
                    batch = []

                if args.limit_entities and n_seen >= args.limit_entities:
                    break

            if batch:
                process_batch(batch, match_writer, cand_writer)

    t1 = time.time()
    print(f'[shard {args.shard_index}] Done in {t1 - t0:.2f}s! Processed {n_seen} entities. Peak RSS: {peak_rss_mb:.1f} MB')

if __name__ == '__main__':
    main()
