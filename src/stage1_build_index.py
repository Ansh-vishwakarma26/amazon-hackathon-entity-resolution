"""
stage1_build_index.py

Build the blocking index + SQLite record store over Source2/Source3.

Training mode uses the original in-memory BlockingIndex and can use
a reduced training index.

Test mode uses DiskBlockingIndex so the full test index does not
consume several GB of RAM.

Run training:

    python src/stage1_build_index.py --data-dir ..\data \
        --ckpt-dir checkpoints \
        --limit-training-entities 200000

Run full test index:

    python src/stage1_build_index.py --data-dir ..\data \
        --ckpt-dir checkpoints_test \
        --source2-file test_source2.tsv \
        --source3-file test_source3.tsv
"""

import argparse
import csv
import os
import pickle
import time
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__)
    )
)

from normalize import (
    normalize_name,
    normalize_address,
    strip_entity_prefix,
)

from blocking import (
    BlockingIndex,
    DiskBlockingIndex,
)

from store import RecordStore


BACKGROUND_MODULO = 10


def load_training_match_ids(
    gt_path,
    limit_entities
):

    selected_s1 = set()
    true_s2 = set()
    true_s3 = set()

    with open(
        gt_path,
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            if (
                len(selected_s1)
                >= limit_entities
            ):
                break

            selected_s1.add(s1_id)

            matches = (
                row.get(
                    "matched_entity_ids",
                    ""
                )
                or ""
            )

            if matches:

                for mid in matches.split(","):

                    mid = mid.strip()

                    if not mid:
                        continue

                    if mid.startswith("S2-"):
                        true_s2.add(
                            strip_entity_prefix(mid)
                        )

                    elif mid.startswith("S3-"):
                        true_s3.add(
                            strip_entity_prefix(mid)
                        )

    return (
        selected_s1,
        true_s2,
        true_s3
    )


def should_keep_background(
    row_number
):

    return (
        row_number
        % BACKGROUND_MODULO
        == 0
    )


def ingest_source(
    path,
    src,
    store,
    index,
    true_ids=None,
    reduced_training=False,
):

    true_ids = true_ids or set()

    total = 0
    kept = 0
    background_kept = 0
    true_kept = 0

    t0 = time.time()

    with open(
        path,
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            total += 1

            iid = strip_entity_prefix(
                row["entity_id"]
            )

            is_true = iid in true_ids

            if reduced_training:

                if is_true:

                    keep = True
                    true_kept += 1

                elif should_keep_background(
                    total
                ):

                    keep = True
                    background_kept += 1

                else:
                    continue

            else:

                keep = True

            if not keep:
                continue

            nn = normalize_name(
                row.get(
                    "business_name",
                    ""
                )
            )

            na = normalize_address(
                row.get(
                    "business_address",
                    ""
                )
            )

            country = (
                row.get(
                    "country",
                    ""
                )
                or ""
            )

            index.add(
                (src, iid),
                nn,
                na
            )

            store.add(
                src,
                iid,
                nn,
                na,
                country
            )

            kept += 1

            if total % 500000 == 0:

                print(
                    f"  ...{src}: "
                    f"{total:,} rows scanned, "
                    f"{kept:,} kept "
                    f"({time.time() - t0:.0f}s elapsed)"
                )

    print(
        f"  {src}: "
        f"{total:,} total rows, "
        f"{kept:,} kept"
    )

    if reduced_training:

        print(
            f"      true matches kept: "
            f"{true_kept:,}; "
            f"background kept: "
            f"{background_kept:,}"
        )

    return total, kept


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        default="dataset/train"
    )

    ap.add_argument(
        "--ckpt-dir",
        default="checkpoints"
    )

    ap.add_argument(
        "--source2-file",
        default="train_source2.tsv"
    )

    ap.add_argument(
        "--source3-file",
        default="train_source3.tsv"
    )

    ap.add_argument(
        "--limit-training-entities",
        type=int,
        default=0,
        help=(
            "Training only: number of "
            "Source1 entities to use. "
            "0 = full data."
        )
    )

    ap.add_argument(
        "--force",
        action="store_true"
    )

    args = ap.parse_args()

    os.makedirs(
        args.ckpt_dir,
        exist_ok=True
    )

    is_test_mode = (
        args.source2_file.startswith("test_")
        and
        args.source3_file.startswith("test_")
    )

    reduced_training = (
        args.limit_training_entities > 0
        and args.source2_file.startswith("train_")
        and args.source3_file.startswith("train_")
    )

    done_marker = os.path.join(
        args.ckpt_dir,
        "stage1.done"
    )

    if (
        os.path.exists(done_marker)
        and not args.force
    ):

        print(
            f"[SKIP] Stage 1 already complete "
            f"(found {done_marker}). "
            f"Use --force to rebuild."
        )

        return

    # ---------------------------------------------------------
    # Training uses the existing pickle index.
    # Test uses the new disk-backed index.
    # ---------------------------------------------------------

    if is_test_mode:

        index_path = os.path.join(
            args.ckpt_dir,
            "blocking_index.db"
        )

    else:

        index_path = os.path.join(
            args.ckpt_dir,
            "index.pkl"
        )

    records_db_path = os.path.join(
        args.ckpt_dir,
        "records.db"
    )

    true_s2 = set()
    true_s3 = set()

    if reduced_training:

        gt_path = os.path.join(
            args.data_dir,
            "train_ground_truth.tsv"
        )

        print(
            f"Reduced training mode: "
            f"first "
            f"{args.limit_training_entities:,} "
            f"Source1 entities"
        )

        print(
            "Loading ground-truth match IDs..."
        )

        (
            selected_s1,
            true_s2,
            true_s3
        ) = load_training_match_ids(
            gt_path,
            args.limit_training_entities
        )

        print(
            f"  selected Source1 entities: "
            f"{len(selected_s1):,}"
        )

        print(
            f"  required S2 true matches: "
            f"{len(true_s2):,}"
        )

        print(
            f"  required S3 true matches: "
            f"{len(true_s3):,}"
        )

        print(
            f"  background sampling: "
            f"approximately "
            f"1/{BACKGROUND_MODULO} records"
        )

    elif is_test_mode:

        print(
            "Full TEST index mode."
        )

        print(
            "Using disk-backed SQLite "
            "blocking index."
        )

    else:

        print(
            "Full index mode."
        )

    store = RecordStore(
        records_db_path,
        fresh=True
    )

    if is_test_mode:

        index = DiskBlockingIndex(
            index_path,
            fresh=True
        )

    else:

        index = BlockingIndex()

    t0 = time.time()

    source_configs = [
        (
            "S2",
            args.source2_file,
            true_s2
        ),
        (
            "S3",
            args.source3_file,
            true_s3
        ),
    ]

    for (
        src,
        fname,
        true_ids
    ) in source_configs:

        path = os.path.join(
            args.data_dir,
            fname
        )

        ingest_source(
            path=path,
            src=src,
            store=store,
            index=index,
            true_ids=true_ids,
            reduced_training=reduced_training
        )

    print(
        "Finalizing blocking index..."
    )

    index.finalize()

    print(
        "Building SQLite record index..."
    )

    store.build_index()
    store.close()

    elapsed = time.time() - t0

    stats = index.stats()

    with open(
        done_marker,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"completed in {elapsed:.0f}s\n"
        )

        f.write(
            f"mode="
            f"{'test_disk' if is_test_mode else 'training'}\n"
        )

        if reduced_training:

            f.write(
                "reduced training index\n"
            )

            f.write(
                f"limit_training_entities="
                f"{args.limit_training_entities}\n"
            )

        f.write(
            f"stats: {stats}\n"
        )

    print()
    print(
        f"Stage 1 done in "
        f"{elapsed:.0f}s."
    )

    print(
        f"Blocking index: "
        f"{index_path}"
    )

    print(
        f"Store: "
        f"{records_db_path}"
    )

    print(
        f"Index stats: "
        f"{stats}"
    )


if __name__ == "__main__":
    main()