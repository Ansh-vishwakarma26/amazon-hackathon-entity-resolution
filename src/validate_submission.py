"""
validate_submission.py
Local validator mirroring the challenge's stated output rules (we don't
have their official utils/validate_submission.py, so this reimplements
every rule from the guidelines PDF). Run this before every leaderboard
upload.

Checks:
  - both files are tab-separated with the correct headers
  - every Source1 TEST entity appears in matching_results.tsv exactly once
  - every Source1 TEST entity appears in candidate_pairs.tsv exactly once
  - no duplicate source1_entity_id rows
  - matched_entity_ids / candidate_entity_ids only reference S2-/S3- ids
    that actually exist in the test set (no S1 self-refs, no unknown ids)
  - no duplicate entity IDs within a single row's list
  - every matched_entity_ids entry appears in that same entity's
    candidate_entity_ids (matches are a subset of candidates)

Run:
    python3 src/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""
import argparse
import csv
import os
import sys


def load_valid_test_ids(test_dir):
    s1_ids, s2_ids, s3_ids = set(), set(), set()
    for fname, target in [("test_source1.tsv", s1_ids),
                           ("test_source2.tsv", s2_ids),
                           ("test_source3.tsv", s3_ids)]:
        path = os.path.join(test_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping that source's ID check.")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                target.add(row["entity_id"])
    return s1_ids, s2_ids, s3_ids


def check_file(path, id_col, list_col, s1_ids, valid_match_ids, issues, label):
    if not os.path.exists(path):
        issues.append(f"[{label}] File not found: {path}")
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != [id_col, list_col]:
            issues.append(f"[{label}] Header mismatch: expected {[id_col, list_col]}, got {reader.fieldnames}")
        seen_ids = set()
        rows_by_id = {}
        for i, row in enumerate(reader):
            sid = row.get(id_col) or row.get('entity_id') or row.get('source1_entity_id') or list(row.values())[0]
            if sid in seen_ids:
                issues.append(f"[{label}] Duplicate {id_col} row: {sid}")
            seen_ids.add(sid)

            if sid.startswith("S1-") and sid not in s1_ids:
                issues.append(f"[{label}] Unknown source1_entity_id not in test set: {sid}")

            raw_list = row[list_col]
            entries = raw_list.split(",") if raw_list else []
            if len(entries) != len(set(entries)):
                issues.append(f"[{label}] Duplicate ids within list for {sid}: {raw_list}")
            for e in entries:
                if e.startswith("S1-"):
                    issues.append(f"[{label}] {sid}: list references a Source1 id ({e}) -- not allowed")
                elif e not in valid_match_ids:
                    issues.append(f"[{label}] {sid}: references unknown/non-test id {e}")
            rows_by_id[sid] = set(entries)

        missing = s1_ids - seen_ids
        if missing:
            issues.append(f"[{label}] {len(missing)} Source1 test entities missing from this file "
                          f"(e.g. {list(missing)[:5]})")
        return rows_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default='output/candidate_pairs.tsv')
    ap.add_argument("--candidate", default="output/candidate_pairs.tsv")
    ap.add_argument("--test-dir", default="dataset/test")
    args = ap.parse_args()

    print("Loading valid test-set IDs...")
    s1_ids, s2_ids, s3_ids = load_valid_test_ids(args.test_dir)
    valid_match_ids = s2_ids | s3_ids
    print(f"  Source1: {len(s1_ids)}, Source2: {len(s2_ids)}, Source3: {len(s3_ids)}")

    issues = []
    print("Checking candidate_pairs.tsv...")
    cand_rows = check_file(args.candidate, "source1_entity_id", "candidate_entity_ids",
                            s1_ids, valid_match_ids, issues, "candidate_pairs")
    print("Checking matching_results.tsv...")
    match_rows = check_file(args.matching, "source1_entity_id", "matched_entity_ids",
                             s1_ids, valid_match_ids, issues, "matching_results")

    if cand_rows is not None and match_rows is not None:
        print("Checking matches are a subset of candidates...")
        for sid, matched in match_rows.items():
            cands = cand_rows.get(sid, set())
            not_subset = matched - cands
            if not_subset:
                issues.append(f"[subset check] {sid}: matched ids not in its candidate list: {not_subset}")

    print()
    if issues:
        print(f"FAIL -- {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues[:50]):
            print(f"  {i+1}. {issue}")
        if len(issues) > 50:
            print(f"  ...and {len(issues) - 50} more")
        sys.exit(1)
    else:
        print("PASS -- no issues found. Safe to submit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
