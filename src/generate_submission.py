"""
generate_submission.py -- Amazon ML Challenge 2026

Generates:
  1. output/candidate_pairs.tsv  (Blocking output)
  2. output/matching_results.tsv (Final ML match predictions)

Meets all 1,732,544 entity requirements, macro F0.5 precision tuning, and TSV standards.
"""

import argparse
import csv
import gc
import os
import pickle
import sys
import time
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address, name_tokens, address_tokens
from features import FEATURE_NAMES

MAX_CANDIDATES_PER_ENTITY = 150


def fmt_key(key):
    return f"{key[0]}-{key[1]}"


def compute_pair_features(s1_n, s1_a, s1_c, cand_n, cand_a, cand_c, cos_sim):
    """Computes similarity metrics using RapidFuzz C-extensions."""
    len_a, len_b = len(s1_n), len(cand_n)
    len_diff_ratio = 1.0 - (abs(len_a - len_b) / max(len_a, len_b)) if max(len_a, len_b) else 1.0
    
    s1_c_norm = (s1_c or "").strip().lower()
    cand_c_norm = (cand_c or "").strip().lower()

    s1_n_tok, cand_n_tok = name_tokens(s1_n), name_tokens(cand_n)
    s1_a_tok, cand_a_tok = address_tokens(s1_a), address_tokens(cand_a)

    n_union = s1_n_tok | cand_n_tok
    name_jaccard = len(s1_n_tok & cand_n_tok) / len(n_union) if n_union else 0.0

    a_union = s1_a_tok | cand_a_tok
    addr_jaccard = len(s1_a_tok & cand_a_tok) / len(a_union) if a_union else 0.0

    s1_num = frozenset(t for t in s1_a.split() if t.isdigit())
    cand_num = frozenset(t for t in cand_a.split() if t.isdigit())
    num_union = s1_num | cand_num
    addr_num_jaccard = len(s1_num & cand_num) / len(num_union) if num_union else 1.0

    ca, cb = s1_n.replace(" ", ""), cand_n.replace(" ", "")
    if ca == cb and ca != "":
        compact_cont = 1.0
    elif ca in cb or cb in ca:
        shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        compact_cont = len(shorter) / len(longer) if len(longer) > 0 else 0.0
    else:
        compact_cont = fuzz.ratio(ca, cb) / 100.0

    return {
        "name_ratio": fuzz.ratio(s1_n, cand_n) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(s1_n, cand_n) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(s1_n, cand_n) / 100.0,
        "name_jaccard": name_jaccard,
        "name_len_diff_ratio": len_diff_ratio,
        "name_compact_containment": compact_cont,
        "name_tfidf_cosine": float(cos_sim),
        "addr_ratio": fuzz.ratio(s1_a, cand_a) / 100.0,
        "addr_jaccard": addr_jaccard,
        "addr_numeric_jaccard": addr_num_jaccard,
        "addr_tfidf_cosine": float(cos_sim),
        "country_match": float(bool(s1_c_norm) and s1_c_norm == cand_c_norm),
        "country_known_both": float(bool(s1_c_norm) and bool(cand_c_norm)),
        "first_char_match": float(bool(s1_n) and bool(cand_n) and s1_n[0] == cand_n[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=r"..\data")
    parser.add_argument("--model-ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Amazon ML Challenge 2026: Submission File Generator ===")
    
    # 1. Load Trained Model and Precision-Tuned Threshold
    model_path = os.path.join(args.model_ckpt_dir, "model", "model.pkl")
    thresh_path = os.path.join(args.model_ckpt_dir, "model", "threshold.txt")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(thresh_path, "r") as f:
        threshold = float(f.read().strip())
    print(f"Loaded classifier. F0.5 Precision Decision Threshold = {threshold:.4f}")

    # 2. Ingest Source 2 & Source 3 Test Data
    print("\n[Step 1/3] Ingesting candidate repositories (Source2 & Source3)...")
    cand_keys = []
    cand_records = []
    cand_corpus = []

    t0 = time.time()
    for src, filename in [("S2", "test_source2.tsv"), ("S3", "test_source3.tsv")]:
        path = os.path.join(args.test_dir, filename)
        if not os.path.exists(path):
            print(f"ERROR: Dataset file missing: {path}")
            sys.exit(1)

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = int(row["entity_id"].split("-")[1])
                nn = normalize_name(row.get("business_name", ""))
                na = normalize_address(row.get("business_address", ""))
                country = row.get("country", "") or ""

                cand_keys.append((src, eid))
                cand_records.append((nn, na, country))
                cand_corpus.append(f"{nn} {na}".strip())

    n_cands = len(cand_keys)
    print(f"  Loaded {n_cands:,} candidate records in {time.time() - t0:.1f}s")

    # 3. Sparse Char-Ngram Vectorization
    print("\n[Step 2/3] Constructing Sparse Vector Index...")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=30000, lowercase=False)

    sample_size = min(300000, n_cands)
    vectorizer.fit(cand_corpus[:sample_size])

    chunk_size = 500000
    sparse_chunks = []
    for i in range(0, n_cands, chunk_size):
        chunk = cand_corpus[i:i + chunk_size]
        sparse_chunks.append(vectorizer.transform(chunk))

    X_cand = vstack(sparse_chunks, format="csr")
    del sparse_chunks, cand_corpus
    gc.collect()

    # 4. Stream Source 1 Queries & Write TSV Outputs
    print("\n[Step 3/3] Generating Candidate & Match Predictions for Source1...")
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    match_file = os.path.join(args.out_dir, "matching_results.tsv")
    cand_file = os.path.join(args.out_dir, "candidate_pairs.tsv")

    s1_rows = []
    s1_corpus = []

    with open(s1_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            nn = normalize_name(row.get("business_name", ""))
            na = normalize_address(row.get("business_address", ""))
            country = row.get("country", "") or ""
            s1_rows.append((eid, nn, na, country))
            s1_corpus.append(f"{nn} {na}".strip())

    n_s1 = len(s1_rows)
    print(f"  Source 1 entities to score: {n_s1:,}")

    t_start = time.time()
    processed_count = 0

    with open(match_file, "w", newline="", encoding="utf-8") as fm, \
         open(cand_file, "w", newline="", encoding="utf-8") as fc:

        writer_m = csv.writer(fm, delimiter="\t")
        writer_c = csv.writer(fc, delimiter="\t")
        writer_m.writerow(["source1_entity_id", "matched_entity_ids"])
        writer_c.writerow(["source1_entity_id", "candidate_entity_ids"])

        for b_start in range(0, n_s1, args.batch_size):
            b_end = min(b_start + args.batch_size, n_s1)
            batch_s1 = s1_rows[b_start:b_end]
            batch_texts = s1_corpus[b_start:b_end]

            X_s1_batch = vectorizer.transform(batch_texts)
            S_sim = X_s1_batch.dot(X_cand.T)

            for idx in range(len(batch_s1)):
                orig_id, s1_n, s1_a, s1_c = batch_s1[idx]
                row_scores = S_sim[idx].toarray().ravel()

                if len(row_scores) > MAX_CANDIDATES_PER_ENTITY:
                    top_indices = np.argpartition(row_scores, -MAX_CANDIDATES_PER_ENTITY)[-MAX_CANDIDATES_PER_ENTITY:]
                    top_indices = top_indices[np.argsort(-row_scores[top_indices])]
                else:
                    top_indices = np.argsort(-row_scores)

                valid_indices = [i for i in top_indices if row_scores[i] > 0.05]
                if not valid_indices and len(top_indices) > 0:
                    valid_indices = top_indices[:10]

                selected_cands = [cand_keys[i] for i in valid_indices]
                writer_c.writerow([orig_id, ",".join(fmt_key(k) for k in selected_cands)])

                if valid_indices:
                    pair_inputs = []
                    for i in valid_indices:
                        cand_n, cand_a, cand_c = cand_records[i]
                        cos_sim = row_scores[i]
                        feat = compute_pair_features(s1_n, s1_a, s1_c, cand_n, cand_a, cand_c, cos_sim)
                        pair_inputs.append([feat[col] for col in FEATURE_NAMES])

                    X_feat = np.array(pair_inputs, dtype=np.float32)
                    probs = model.predict_proba(X_feat)[:, 1]

                    matched = [fmt_key(selected_cands[k]) for k in range(len(probs)) if probs[k] >= threshold]
                    writer_m.writerow([orig_id, ",".join(matched)])
                else:
                    writer_m.writerow([orig_id, ""])

                processed_count += 1

            elapsed = time.time() - t_start
            rate = processed_count / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed_count:,}/{n_s1:,} entities ({elapsed:.0f}s elapsed, {rate:.1f} entities/s)")

    print(f"\nExecution Complete! Total processing time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()