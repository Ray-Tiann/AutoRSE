"""Generate a label-free popularity submission for the public schema."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
runpy.run_path(str(Path(__file__).resolve().parent / "baseline/popularity_submission.py"), run_name="__main__")
