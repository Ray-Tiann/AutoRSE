# Maintainer setup: reviewed submissions to Hugging Face

This pipeline accepts one public submission JSON per PR, verifies signed evaluator receipts and pinned evidence, requires qualified human GitHub approval, and publishes reviewed results to the existing Static Space. It does not launch research runs or deploy a scoring server.

## Bootstrap the trusted source

Install these files in the root of `Ray-Tiann/AutoRSE` through a separate maintenance change:

- `.github/workflows/validate-leaderboard.yml` and `publish-leaderboard.yml`.
- `leaderboard/autorse_leaderboard/`, `requirements.txt`, public `trusted-keys.json`, tests and these guides.
- `leaderboard/submissions/` containing only its README or `.gitkeep` until a real result is approved. Keep the incomplete template outside this directory.

Review the validator, publisher, dependency pins, workflows and evaluator public keys before merging. The trusted key file contains **public verification keys only**. Keep each evaluator's private signing key in the scoring service's secret configuration; do not put private keys, labels or account credentials in GitHub or the Space. A receipt from an unknown evaluator key must fail validation.

Use an independent maintenance test workflow for source changes. The submission publisher listens only to `leaderboard/submissions/**` on `main` and to an explicit manual run, so a code-only bootstrap commit does not publish results.

## Configure the Space's trusted publisher

In [Ray0311/AutoRSE-Leaderboard settings](https://huggingface.co/spaces/Ray0311/AutoRSE-Leaderboard/settings), open **Trusted Publishers** and add a GitHub Actions publisher with all three claims:

| Claim | Value |
| --- | --- |
| Repository | `Ray-Tiann/AutoRSE` |
| Branch | `main` |
| Workflow | `publish-leaderboard.yml` |

Use the workflow filename, not its display name. This workflow does not specify a GitHub environment. Its `HF_OIDC_RESOURCE` is `spaces/Ray0311/AutoRSE-Leaderboard`. Do not widen the trust rule to all branches or unrelated workflows.

Hugging Face exchanges the GitHub OIDC identity for a short-lived, repository-scoped credential. `huggingface-hub==2.3.0`, pinned in the requirements, handles this exchange. No long-lived `HF_TOKEN` repository secret or interactive login is required. Only the main publisher job has `id-token: write`; the PR validator has no OIDC permission. See the [official Trusted Publishers documentation](https://huggingface.co/docs/hub/trusted-publishers).

The Space frontend must read `community-leaderboard.json` for reviewed results and display the paper's historical reference data separately. The publisher also uploads `approval-ledger.json` and `submissions/<id>.json`. Its live connection check writes only `publisher-status.json`, containing the source commit and protocol status, without scores. It does not overwrite `index.html` or the historical paper data. An empty reviewed cohort must be shown honestly as empty; do not populate it with test receipts or transplant historical scores into the public-data cohort.

## Protect the review boundary

Protect `main` against direct participant pushes and require PR review. Enable dismissal of stale approvals and require the latest push to be reviewed. Assign trusted maintainers to workflow, validator, publisher, dependency and key changes through repository review policy/CODEOWNERS. Changes to those components are maintenance changes, separate from submission PRs.

The submission validation workflow is path-filtered. **Do not make its status globally required on every PR**: GitHub can leave a required path-skipped check pending on unrelated maintenance changes. Require successful submission validation as part of submission review policy, and use an always-running maintenance check for globally required source CI. The post-merge publisher independently enforces signatures, evidence and GitHub approval even if an administrator bypasses a branch rule.

Eligible submission PRs add exactly one `leaderboard/submissions/<id>.json`, with a matching ID and a maximum size of 2 MiB. Mixed code/submission PRs, modifications, renames, nested paths and invalid slugs fail. Code-only PRs do not trigger this submission workflow and do not receive a submission eligibility claim.

The validator uses `pull_request_target` only to run the explicitly checked-out **base commit**. It has `contents: read` and `pull-requests: read`, does not expose repository secrets or an OIDC token, disables persisted checkout credentials and never checks out or executes PR code. It fetches only the JSON blob through GitHub REST and treats all linked artifacts as bytes. Keep these properties intact: GitHub warns that executing untrusted PR code with `pull_request_target` can expose credentials. See [the event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target) and [secure use reference](https://docs.github.com/en/actions/reference/security/secure-use#mitigating-the-risks-of-untrusted-code-checkout).

After merge, `github_review collect` queries GitHub for the PR and its paginated reviews, verifies a merged main-branch PR adding the exact current bytes, and accepts only a current-head human approval by a non-author with current `write`, `maintain` or `admin` permission. A reviewer's latest approval/change-request/dismissal controls that reviewer's decision; later comments alone do not revoke approval. The ledger records the author, PR/head/merge SHAs, reviewer evidence and submission SHA256. GitHub API failures stop publication rather than assuming approval. The read-only Actions token must be able to query PR reviews and repository metadata; the workflow does not need GitHub write permission. [GitHub documents the reviewer-permission endpoint here](https://docs.github.com/en/rest/collaborators/collaborators#get-repository-permissions-for-a-user).

## What publication verifies

The automatic pipeline checks three complete signed six-task benchmark receipts, 18 unique task/trial records, distinct per-task seeds, pinned public-data protocol, one-GPU/12-hour declarations, timestamp order and public evidence bytes. Structured configuration and provenance must agree with the submission and signed prediction digests. The validator never executes participant commands or source archives.

A maintainer must assess whether those declarations are true: actual runtime and resources, independent trials, complete trajectories, evaluated source reproducibility, hidden-label access and score manipulation. Signed scores prove what the evaluator scored, not that the preceding experiment followed every research rule. Test/demo receipts remain ineligible regardless of approval.

This release accepts only `autorse-public-v2-20261029` and its fixed bundle SHA256. Preserve the framework commit and exact model/agent versions in each entry. If datasets, evaluation protocol or execution environments change in ways that affect comparability, introduce a new protocol revision/cohort and review that change before accepting results. Do not silently pool incompatible versions or historical paper runs.

The publisher rechecks **every** approved JSON and its public, commit-pinned evidence before producing a snapshot. Any unapproved/test entry, altered approved bytes, failed receipt, unavailable artifact or integrity mismatch aborts the whole upload, preserving the existing published snapshot. Evidence lives in submitters' datasets; the publisher mirrors the JSON records and review ledger, not the underlying evidence archives. Public availability is therefore an ongoing dependency.

Main-tip checks at collection and immediately before upload reject an older GitHub snapshot after `main` advances. Publisher runs share a single concurrency group with cancellation disabled. The Hugging Face commit uses its previous commit as `parent_commit`, so a concurrent Space edit causes a conflict instead of silently overwriting a newer snapshot. Rerun the publisher from current `main` after resolving a failure.

## Test before the first real result

From the repository root, run the local test suite with Python 3.12:

```bash
python -m venv .venv-leaderboard
source .venv-leaderboard/bin/activate
python -m pip install -r leaderboard/requirements.txt
PYTHONPATH=leaderboard python -m unittest discover -s leaderboard/tests -v
```

Review tests use synthetic fixtures and temporary local Git history. They exercise JSON-only PR extraction, approval/reviewer state, current-head and main-tip checks, file hashes and fail-closed behavior; these fixtures are not publishable submissions.

After merging the trusted workflows and configuring the HF Trusted Publisher, open **Actions → Publish reviewed leaderboard → Run workflow** and select `main`. This is a real end-to-end OIDC permission check even when there are no approved results. The workflow collects the review ledger, then runs `check-publisher` before the score publication step:

```bash
PYTHONPATH=leaderboard python -m autorse_leaderboard.cli check-publisher \
  --repo-id Ray0311/AutoRSE-Leaderboard \
  --sha "$GITHUB_SHA"
```

Run this through GitHub Actions so its short-lived OIDC identity is available. The command checks the current GitHub main commit and the fixed HF destination, and writes only `publisher-status.json` to the Space. Confirm that the Actions step succeeds and that this file appears at the expected HF commit. This verifies the live authentication and write path without adding simulated results, changing the leaderboard data or claiming that any benchmark entry has been approved. With an empty reviewed cohort, score publication is still skipped. A permission, trust-claim or stale-main failure must remain a failed check and be corrected before inviting submissions.

Once a genuine benchmark submission is merged and approved, use a clean checkout of current `main` to inspect the generated snapshot without uploading. Use an already authenticated read-only GitHub session for `GITHUB_TOKEN`; do not paste a token into command history or source files. Set `GITHUB_REPOSITORY=Ray-Tiann/AutoRSE` and `GITHUB_SHA` to the full checked-out main commit, then run:

```bash
PYTHONPATH=leaderboard python -m autorse_leaderboard.github_review collect \
  --repository "$GITHUB_REPOSITORY" \
  --sha "$GITHUB_SHA" \
  --submissions-dir leaderboard/submissions \
  --approved-dir /tmp/autorse-approved-submissions \
  --output /tmp/autorse-approval-ledger.json

PYTHONPATH=leaderboard python -m autorse_leaderboard.cli publish \
  --submissions-dir /tmp/autorse-approved-submissions \
  --keys leaderboard/trusted-keys.json \
  --approval-ledger /tmp/autorse-approval-ledger.json \
  --output-dir /tmp/autorse-leaderboard-preview \
  --repo-id Ray0311/AutoRSE-Leaderboard \
  --dry-run
```

Use a new empty output path for `--approved-dir` each time. The collector refuses to overwrite an existing directory. Do not manufacture an approval ledger for a live publication. With zero submission JSONs, collection returns an empty ledger and Actions skips the publish step. The direct `publish` command refuses an empty source directory.

After a real merge, check the **Publish reviewed leaderboard** Actions run, its saved review/output artifacts and the new Space commit. Publication artifacts are retained for 90 days and validation reports for 30 days; the Space's review ledger and JSON snapshot provide the public publication record. A failing check or missing Trusted Publisher configuration is not a published result.

## Workflow dependency pins

Actions are pinned to verified upstream commit SHAs, rather than mutable release tags:

| Action | Verified release | Commit |
| --- | --- | --- |
| `actions/checkout` | v6.0.2 | [`de0fac2e4500dabe0009e67214ff5f5447ce83dd`](https://github.com/actions/checkout/commit/de0fac2e4500dabe0009e67214ff5f5447ce83dd) |
| `actions/setup-python` | v7.0.0 | [`5fda3b95a4ea91299a34e894583c3862153e4b97`](https://github.com/actions/setup-python/commit/5fda3b95a4ea91299a34e894583c3862153e4b97) |
| `actions/upload-artifact` | v7.0.2 | [`cf430e030ddbb5b0abf93d22962f4752f3646cd9`](https://github.com/actions/upload-artifact/commit/cf430e030ddbb5b0abf93d22962f4752f3646cd9) |

Review dependency upgrades through the trusted maintenance path. No account authentication or trusted-publisher configuration is created automatically by these files.
