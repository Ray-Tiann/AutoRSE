# Task contract

Retain WuKong FMB/LCB feature interactions. Fit feature processors only on training rows and preserve their artifacts with the checkpoint. Select by full public-validation AUC. Return sample_id,score in a gzip CSV in public test order.

From this environment directory (`/workspace` in the task container), run the
unchanged starter on an experiment branch:

```bash
python3 -m pip install -r requirements.txt
python3 baseline_git.py init
python3 baseline_git.py new exp/001-baseline --from baseline
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh movielenslatest
python3 baseline_git.py record --validation 'PUBLIC VALIDATION RESULT' \
  --notes 'Unchanged starter; describe the observed outcome'
```

Replace the validation text with the result actually measured on public validation.
Public inputs belong in `datasets/movielenslatest/input/`; the run writes
`datasets/movielenslatest/working/submission.csv.gz`. Hidden labels are not distributed.
After a successful run, save its prediction outside Git before another experiment
can overwrite it. For example:

```bash
mkdir -p datasets/movielenslatest/working/experiments/exp/001-baseline
cp datasets/movielenslatest/working/submission.csv.gz \
  datasets/movielenslatest/working/experiments/exp/001-baseline/submission.csv.gz
```

Use a unique branch/run snapshot path for each candidate you might select, and
keep a good snapshot after a later failure.

The runtime initializes `baseline/.git` on branch `baseline` with the immutable
`autorse-baseline` tag. The explicit `init` above is safe to repeat and does not
reset existing history. For each coherent experimental change, create a branch
**before editing**, then commit all changed source, helpers, and configuration
under `baseline/` before running:

```bash
python3 baseline_git.py new exp/002-description --from exp/001-baseline
# Edit source, helpers, and configuration inside baseline/.
python3 baseline_git.py commit -m 'Describe this experimental change'
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh movielenslatest
python3 baseline_git.py record --validation 'PUBLIC VALIDATION RESULT' \
  --notes 'Change, outcome, and selection rationale'
```

Use one branch per logical experiment, not per file save. Start independent ideas
with `--from baseline` or build on a prior `exp/...` branch. Keep failed branches.
The run wrapper records the branch, evaluated code commit, command, exit
status, and prediction SHA256 in `.autorse-runs/` outside Git. After every real
run, use `record` to commit its metadata under `baseline/experiments/`, including
failures: state that validation is unavailable when appropriate and describe the
error in `--notes`. Never substitute hidden-test feedback for public validation.

Keep experiment source, helpers, and configuration inside `baseline/` so the Git
history captures them. Reserve `baseline/experiments/` for generated JSON run
records, and put model code/configuration elsewhere under `baseline/`. Keep the
workspace wrapper and requirements unchanged; the exporter declares and hashes required runtime files outside the baseline.
Do not push the runtime repository, use submodules, or add its nested `.git`
directory to the public source checkout.

Before finalization, switch to the branch selected by public validation, inspect
its status, and ensure all source and experiment records are committed:

```bash
git -C baseline switch exp/001-baseline  # Replace with your selected branch.
git -C baseline status --short
# Restore the snapshot for the selected branch/run.
cp datasets/movielenslatest/working/experiments/exp/001-baseline/submission.csv.gz \
  datasets/movielenslatest/working/submission.csv.gz
```

The status must be empty. Restore the selected candidate's exact saved prediction;
branch switching does not restore files outside Git. Export requires a recorded
successful run on the selected branch whose code/ancestry, required runtime files,
and prediction bytes match. An earlier best candidate can be exported after
later failed experiments when these checks pass. If no matching prediction/receipt
is available, rerun the selected branch and record that run before exporting.
Then export from this environment directory:

```bash
python3 export_artifact.py \
  --command 'CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh movielenslatest' \
  --validation 'PUBLIC VALIDATION RESULT AND SELECTION RATIONALE'
```

Include any nondefault environment settings and arguments in `--command`.
Use `--overwrite` to replace a prior selected artifact. The exporter requires a
clean Git repository and matching recorded successful run/prediction provenance.
`/workspace/artifact/` contains the selected code and configuration, `README.md`, `GIT_PROVENANCE.json`,
`baseline.git.bundle` with all experiment branches and code history, and
`SHA256SUMS`. The bundle retains experiment records, including failed attempts;
weights, input data, caches, and credentials must stay out of Git and the export.

Harbor collects this whole artifact separately from the prediction file. Retain
it for reproduction and code audit. Upload **only the prediction file** to the
hosted evaluator; do not upload the bundle to the public Hugging Face Dataset or
Space. The local verifier checks format only; hidden-test scores come from the
hosted evaluator and must not guide experiment selection.
