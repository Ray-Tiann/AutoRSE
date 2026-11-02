# Maintainer tooling

Running an agent or installing public data does not require these directories.

| Directory | Purpose |
| --- | --- |
| `space/` | Self-hosted CPU evaluator template; the official hosted service is separate. |
| `hf-leaderboard/` | Hugging Face leaderboard page. |
| `leaderboard/` | Submission validation, review checks, and leaderboard publishing. |
| `tests/` | Framework regression tests. |

Run development checks from the repository root:

```bash
python -m unittest discover -s internal/tests -v
PYTHONPATH=internal/leaderboard python -m unittest discover -s internal/leaderboard/tests -v
```
