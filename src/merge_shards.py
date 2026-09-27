"""
merge_shards.py
Combines matching_results_shard{N}.tsv / candidate_pairs_shard{N}.tsv
(produced by running infer.py with --num-shards > 1) into the final
matching_results.tsv / candidate_pairs.tsv the challenge expects.

Row order doesn't matter for the submission rules (every Source1 test
entity needs exactly one row -- nothing says what order), so this simply
writes the header once, then concatenates each shard's data rows in
shard order.

Run (after all shards finish):
    python3 src/merge_shards.py --out-dir output --num-shards 4
"""
import argparse
import os


def merge_one(out_dir, base_name, num_shards):
    final_path = os.path.join(out_dir, f"{base_name}.tsv")
    header_written = False
    total_rows = 0
    with open(final_path, "w", newline="", encoding="utf-8") as out_f:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(out_dir, f"{base_name}_shard{shard_idx}.tsv")
            if not os.path.exists(shard_path):
                print(f"  WARNING: {shard_path} not found -- skipping (did that shard finish?)")
                continue
            with open(shard_path, newline="", encoding="utf-8") as in_f:
                lines = in_f.readlines()
                if not lines:
                    continue
                if not header_written:
                    out_f.write(lines[0])
                    header_written = True
                out_f.writelines(lines[1:])
                total_rows += len(lines) - 1
    print(f"  {final_path}: {total_rows} rows written")
    return total_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    print(f"Merging {args.num_shards} shards from {args.out_dir}/...")
    print("matching_results:")
    n_match = merge_one(args.out_dir, "matching_results", args.num_shards)
    print("candidate_pairs:")
    n_cand = merge_one(args.out_dir, "candidate_pairs", args.num_shards)

    if n_match != n_cand:
        print(f"\nWARNING: row count mismatch -- matching_results has {n_match}, "
              f"candidate_pairs has {n_cand}. Check that all shards completed before merging.")
    else:
        print(f"\nMerged {n_match} rows into both final files. "
              f"Now run validate_submission.py before uploading.")


if __name__ == "__main__":
    main()