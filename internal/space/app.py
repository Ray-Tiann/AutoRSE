"""Hugging Face Space entry point; upload predictions, receive aggregate scores."""
import os
from pathlib import Path
import sqlite3
import time

UPLOAD_ROOT = Path("/tmp/autorse-uploads")
os.environ["GRADIO_TEMP_DIR"] = str(UPLOAD_ROOT)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
from service import PRIVATE_ROOT, EvaluatorUnavailable, SubmissionRejected, provision_private_data, score_submission
from autorse.tasks import TASKS

STATE_ROOT = Path(os.environ.get("AUTORSE_STATE_DIR", "/tmp/autorse-state"))
DAILY_LIMIT = int(os.environ.get("AUTORSE_DAILY_LIMIT", "5"))
READY = False


def reserve_attempt(user: str, task: str) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    day = int(time.time() // 86400)
    with sqlite3.connect(STATE_ROOT / "attempts.sqlite3", timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS attempts (user TEXT, task TEXT, day INTEGER)")
        db.execute("BEGIN IMMEDIATE")
        used = db.execute("SELECT COUNT(*) FROM attempts WHERE user=? AND task=? AND day=?", (user, task, day)).fetchone()[0]
        if used >= DAILY_LIMIT:
            raise SubmissionRejected("Daily submission limit reached for this task.")
        db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (user, task, day))


def submit(task: str, prediction: str | None, profile: gr.OAuthProfile | None):
    if profile is None:
        raise gr.Error("Sign in with Hugging Face before submitting.")
    if not READY:
        raise gr.Error("The evaluator is not configured yet. Contact the benchmark maintainer.")
    if task not in TASKS or prediction is None:
        raise gr.Error("Select a task and upload its prediction file.")
    path = Path(prediction)
    if path.is_symlink() or not path.resolve().is_relative_to(UPLOAD_ROOT.resolve()):
        raise gr.Error("Only uploaded prediction files are accepted.")
    try:
        reserve_attempt(profile.username, task)
        return score_submission(task, path)
    except (SubmissionRejected, EvaluatorUnavailable) as error:
        raise gr.Error(str(error)) from None
    except Exception:
        raise gr.Error("Scoring is temporarily unavailable.") from None


with gr.Blocks(title="AutoRSE Evaluation", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# AutoRSE Evaluation\nUpload **predictions only**. Training and model selection use public validation data. "
                "Hidden labels stay on this server; only aggregate test metrics are returned.")
    gr.LoginButton()
    task = gr.Dropdown(choices=list(TASKS), value="hstu", label="Task")
    prediction = gr.File(label="Prediction (.npz / .csv / .csv.gz)", type="filepath", file_count="single")
    gr.Markdown("Up to 160 MiB per file. Five attempts per account/task/UTC day by default; invalid attempts count. "
                "A hidden-test score alone does not certify protocol compliance or reproduce a research trial.")
    button = gr.Button("Submit for scoring", variant="primary")
    result = gr.JSON(label="Aggregate result")
    button.click(submit, inputs=[task, prediction], outputs=result, api_name="submit", concurrency_limit=1)


if __name__ == "__main__":
    try:
        provision_private_data()
        READY = True
    except EvaluatorUnavailable:
        print("AutoRSE: configure private evaluator repository/revision/secrets before accepting submissions.")
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860, share=False,
                                 max_file_size=160 * 1024**2, show_error=False,
                                 blocked_paths=[str(PRIVATE_ROOT.resolve()), str(STATE_ROOT.resolve())])
