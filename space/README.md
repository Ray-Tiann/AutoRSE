---
title: AutoRSE Evaluation
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
hf_oauth: true
---

# AutoRSE Evaluation

Upload the prediction file for one of the six AutoRSE tasks. Sign in with Hugging Face first.
The service returns aggregate metrics. Hidden labels are fetched only by the server from an
independent **private** dataset repository pinned to a commit. They are never shipped with public inputs.

Maintainer setup: see `docs/HOSTED_EVALUATION.md` in the GitHub repository.
This template is not a deployed scoring endpoint until those settings are configured.
