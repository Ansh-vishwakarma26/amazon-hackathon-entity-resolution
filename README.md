# Business Entity Resolution — Amazon ML Challenge 2026

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
```

Put the real challenge data in place:
```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

## How the pipeline works

Five stages, each **checkpointed to disk** and **safe to rerun** — if your
machine kills a stage (memory, timeout, you just need to stop), rerun the
exact same command and it picks up where it left off instead of redoing
finished work.

```bash
./run_pipeline.sh dataset/train checkpoints
```

Or run stages individually:

```bash
python3 src/stage1_build_index.py --data-dir dataset/train --ckpt-dir checkpoints
python3 src/stage2_generate_pairs.py --data-dir dataset/train --ckpt-dir checkpoints --chunk-size 5000
python3 src/stage3_fit_featurizer.py --ckpt-dir checkpoints
python3 src/stage4_compute_features.py --ckpt-dir checkpoints
python3 src/stage5_train_model.py --ckpt-dir checkpoints --data-dir dataset/train
```

**Stage 1** — builds the blocking index (candidate generation keys) and a
SQLite record store over Source2/Source3. Slowest stage; only runs once
(checkpointed as `checkpoints/index.pkl` + `checkpoints/records.db`).

**Stage 2** — for each Source1 training entity, generates blocking
candidates and labels them against ground truth. Writes chunked files to
`checkpoints/pairs_chunks/`. If this dies partway through, just rerun —
finished chunks (marked with a `.done` file) are skipped.

**Stage 3** — fits the TF-IDF vectorizers on a sample of the text (not
every document — see comment in `stage3_fit_featurizer.py` for why).

**Stage 4** — computes the actual similarity features for each pair chunk
(Levenshtein ratio, token-sort/set ratio, Jaccard, TF-IDF cosine, etc.),
one chunk in memory at a time. This is the stage that OOM'd when we tried
to do it all in one big batch — chunking fixes that.

**Stage 5** — loads all feature chunks, trains the classifier (LightGBM if
installed, else sklearn's HistGradientBoostingClassifier), tunes the
decision threshold to maximize macro F_0.5 (matching the exact scoring
formula, singleton rule included), and saves the model.

## If you run out of memory or time on a stage

- Lower `--chunk-size` in stage 2 (default 5000) — smaller chunks = less
  memory per step, more checkpoint files.
- You do NOT need to redo earlier stages. Only the stage that failed
  needs rerunning, and it picks up from its last completed chunk.

## What's built so far / what's next

Done: `normalize.py`, `blocking.py`, `store.py`, `features.py`, and the
full training pipeline (stages 1-5) — validated end-to-end on a real
800K-record sample of the actual data, achieving **local validation
F_0.5 = 0.929**.

Still to build: `infer.py` (generates `matching_results.tsv` and
`candidate_pairs.tsv` for the actual test set using the trained model),
`utils/validate_submission.py`-equivalent local validator, and the
methodology writeup.

## Performance note for the real 8GB-RAM run

At full scale (1.7M+ Source1 test entities), stage 4's TF-IDF cosine
feature computation is the main cost. Measured here (no rapidfuzz, 1 CPU
core): ~6,100 pairs/sec. With rapidfuzz installed (C-accelerated) this
should be meaningfully faster. If it's still too slow: lower
`MAX_CANDIDATES_PER_ENTITY` in `blocking.py` (currently 150) — fewer
candidates per entity means fewer pairs to score, at a small recall cost.
