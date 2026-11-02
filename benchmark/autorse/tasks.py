"""Fixed task and submission contracts; these contain no hidden labels."""
TASKS = {
    "hstu": {"dataset": "beauty", "file": "submission.npz", "rows": 22363, "metric": "ndcg_at_10",
             "arrays": {"user_ids": (22363,), "top_items": (22363, 10)}},
    "tiger": {"dataset": "beauty", "file": "submission.npz", "rows": 22363, "metric": "ndcg_at_10",
              "arrays": {"user_ids": (22363,), "top_items": (22363, 10)}},
    "intsr": {"dataset": "Amazon", "file": "submission.npz", "rows": 133391, "metric": "macro_ndcg_at_10",
              "arrays": {"sample_ids": (133391,), "task_ids": (133391,),
                         "candidate_items": (133391, 100), "scores": (133391, 100)}},
    "wukong": {"dataset": "movielenslatest", "file": "submission.csv.gz", "rows": 200686, "metric": "auc",
               "columns": ["sample_id", "score"]},
    "rankmixer": {"dataset": "criteo5m", "file": "submission.npz", "rows": 500000, "metric": "auc",
                  "arrays": {"sample_ids": (500000,), "scores": (500000,)}},
    "onetrans": {"dataset": "kuaisar_v2", "file": "submission.csv", "rows": 1460572, "metric": "auc",
                 "columns": ["sample_id", "prediction"]},
}


def task_name(value):
    name = value.removesuffix("_task").lower()
    if name not in TASKS:
        raise ValueError("Unknown task")
    return name
