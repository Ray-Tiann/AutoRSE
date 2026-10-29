"""Public format check; no hidden labels and no recommendation-quality score."""
import json
import os
from pathlib import Path
from autorse.submissions import validate

logs = Path(os.environ.get("LOG_DIR", "/logs/verifier"))
logs.mkdir(parents=True, exist_ok=True)
report = {"evaluation": "format_only_not_official_score", "format_valid": False}
try:
    submission = Path("/logs/artifacts/workspace") / os.environ["SUBMISSION_REL"]
    report.update(validate(os.environ["TASK_NAME"], submission))
    report["format_valid"] = True
except (ValueError, OSError, EOFError, KeyError) as exc:
    report["error"] = str(exc)
(logs / "format_report.json").write_text(json.dumps(report, indent=2) + "\n")
(logs / "reward.json").write_text(json.dumps({"format_valid": float(report["format_valid"])}) + "\n")
print(json.dumps(report))
