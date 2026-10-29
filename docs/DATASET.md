---
pretty_name: AutoRSE public benchmark data — sanitized v2
task_categories:
  - other
tags:
  - recommendation
  - benchmark
  - click-through-rate-prediction
  - tabular
---

# AutoRSE public benchmark data

This repository distributes the agent-visible data for six recommendation research tasks: HSTU, TIGER, IntSR, WuKong, RankMixer, and OneTrans. It contains training data, public validation labels, public catalog/features/history inputs, and unlabeled test requests. Official scoring is intended for the separate [RayTian/AutoRSE evaluation Space](https://huggingface.co/spaces/RayTian/AutoRSE), once the maintainer deploys it. Hidden evaluator labels and reference submissions are not distributed here.

Download `recagent-data-public-v2.zip`, retain `SHA256SUMS`, and use the release's `tools/install_public_data.py` from the benchmark code repository. This is a native benchmark data archive; `datasets.load_dataset()` is not the installation interface. The archive preserves paths below `benchmark/tasks/<task>/environment/datasets/`.

```bash
# Install from a downloaded archive. Use the code repository as --root.
python3 tools/install_public_data.py \
  --archive /path/to/recagent-data-public-v2.zip --root /path/to/AutoRSE

# Or download from the public dataset repository and install one task.
python3 tools/install_public_data.py \
  --repo-id RayTian/AutoRSE --revision main \
  --task wukong --root /path/to/AutoRSE
```

The installer requires Python 3.9 or newer and only its standard library. Its default download source is [RayTian/AutoRSE](https://huggingface.co/datasets/RayTian/AutoRSE); the data must be uploaded before downloads work. It pins the ZIP's size and SHA-256, validates its public-only paths and embedded per-file manifest, rejects symlinks and duplicate ZIP members, and publishes each verified file atomically. Identical existing files are retained; differing files cause an error and are never replaced. Repeat `--task` to select several tasks. A Hugging Face commit revision is recommended for reproducible downloads; the pinned ZIP checksum remains mandatory for every revision.

## Tasks and split identity

| Task | Dataset and supplied protocol | Train | Public validation | Test input |
| --- | --- | ---: | ---: | ---: |
| HSTU | Amazon Reviews 2014 Beauty 5-core; chronological leave-two-out; full-catalog retrieval | 153,776 interactions / 22,363 users | 22,363 queries | 22,363 user IDs |
| TIGER | Same Beauty data and ID mapping as HSTU, with item embeddings and semantic IDs | 153,776 interactions / 22,363 users | 22,363 queries | 22,363 user IDs |
| IntSR | Sanitized v2 Amazon Kindle; recommendation and search candidate ranking, 100 candidates per query | 734,267 recommendation + 716,493 search rows | 51,654 recommendation + 55,853 search rows | 65,168 recommendation + 68,223 search queries |
| WuKong | `MovielensLatest_x1`, RecZoo/BARS preprocessing | 1,404,801 rows | 401,372 rows | 200,686 rows |
| RankMixer | Criteo first 5M; chronological 4M / 500K / 500K | 4,000,000 rows | 500,000 rows | 500,000 rows |
| OneTrans | KuaiSAR v2; chronological recommendation samples with earlier recommendation/search event histories | 11,684,572 rows | 1,460,572 rows | 1,460,572 rows |

Counts are taken from the public archive's metadata and structural inspection. OneTrans also contains a shared table of 19,664,885 events. Beauty has 12,101 catalog items; its embeddings have 768 dimensions and semantic IDs have three components. HSTU and TIGER intentionally keep separate, identical copies because their task runners require separate paths. The Beauty summary JSON also mentions MovieLens-1M and Video Games; those datasets are not included in this release.

The standalone `recagent-data-manifest.json` records the final archive's exact file inventory, byte sizes, and SHA-256 hashes. It is also included in the ZIP. The ZIP is 941,569,295 bytes and expands to 2,105,027,108 bytes across 69 dataset files. Test labels are absent or represented by constant compatibility placeholders; a zero placeholder must not be interpreted as a negative target.

## Sanitized v2 and historical results

The earlier `recagent-data-public-v1.zip` was found to expose held-out outcomes through auxiliary public tables and a reproducible test-candidate permutation. The v1 archive is not distributed in this dataset repository. V2 preserves test-query identity and file paths while removing those routes to the held-out outcomes. **V2 is a new data revision: its IntSR public split sizes, history context, and test candidate order differ from v1, and historical paper scores are not v2 re-evaluations.**

For OneTrans, every test request's `sample_id` identified its current row in the shared event table. V2 sets `click`, `like`, `follow`, `forward`, and `search_action` to zero in all 1,460,572 corresponding event rows, and sets the four outcome columns in `samples_test.parquet` to zero. All other event rows, non-outcome values, table order, column dtypes, history offsets, and training/validation parquet files are preserved. Held-out outcomes remain unavailable even when a test event precedes a later test request. This changes the information available at test time; historical metrics obtained using v1 are not guaranteed to reproduce with v2.

For IntSR, user histories and session-to-positive-item tables could reveal held-out recommendation/search outcomes across both streams. V2 retains only events strictly before each user's earliest test request across recommendation and search. This removes 50,604 labeled train/validation rows and 194,941 history events, zeros 95,264 nonpublic session-to-positive-item entries in both lookup tables, and clamps the test requests' history counts to retained prefixes. Test sample IDs, candidate sets, queries, timestamps, and frozen Qwen feature arrays remain unchanged. Each test row receives a fresh candidate permutation generated from nonpublic cryptographic entropy, and the old permutation-seed metadata is removed. **Predict scores in the candidate order shipped in v2; original v1 candidate-position predictions are incompatible.** Session vocabulary provenance hashes and internal manifests are refreshed. Details and exact split counts are recorded in `INTSR_PUBLIC_DATA_AUDIT.json`.

The release audit reports contain aggregate counts and verification results, never individual answers. No private labels were opened to build this release.

For evaluator maintainers, `intsr-test-identity.npz` contains only the three public arrays `sample_ids`, `task_ids`, and `candidate_items` in v2 order. It has no labels and uses integer arrays readable with `allow_pickle=False`. Its SHA-256 is `2bb3a68d67e2db24631a550a5a4b98b934efae0202d7018e13396ff742dd5757`. Use this identity artifact to bind the evaluator's expected candidate order to v2; private target labels remain separately provisioned. The identity file is outside the 69-file installation ZIP and is not needed for agent training.

These checks address direct target exposure in the supplied artifacts. They do not establish that the original public source datasets are unavailable on the internet, that external retrieval cannot reconstruct a split, or that trained features carry no information about their source corpus. Run evaluated agents with the benchmark's documented data-access rules.

## Formats and safe use

Data files include NumPy arrays, NPZ features, CSV, Parquet, JSON, and legacy Python pickle files required by IntSR. The installer copies these as inert bytes and does not deserialize pickle files or execute scripts. Load legacy pickle artifacts only from this checksum-verified release using the benchmark's trusted readers. Full installation uses approximately 2.1 GB before any training output; download and extraction also need temporary disk space. A task-selective install downloads the whole ZIP and extracts only the selected task paths.

Excluded assets include `tests/private/`, hidden target arrays, `solution/reference_submission.*`, model checkpoints, experiment runs, logs, credentials, and environment files. Training and public validation labels are intentional public inputs. Item semantic IDs, item embeddings, and frozen query features are public model inputs, not hidden evaluator answer files.

## Provenance and licensing

This is a processed compilation of upstream datasets. **No blanket license or new permission to use the upstream data is granted by this repository. Upstream dataset and derived-feature terms continue to apply.** The local source archive does not contain sufficient licensing evidence to assert a single license for the compilation, so its dataset card deliberately does not claim an open-source data license.

| Component | Provenance recorded in the local public artifacts | Licensing evidence still needed from the maintainer |
| --- | --- | --- |
| Beauty (HSTU/TIGER) | Amazon Reviews 2014 Beauty 5-core; `phonism/genrec`, `genrec.data.amazon.AmazonSeqDataset`; bundled public embeddings and RQ-VAE semantic IDs | Confirm the upstream data redistribution terms and the derivation/license of embeddings and semantic IDs. A code license is not a dataset license. |
| IntSR | Processed Amazon Kindle from the IntSR/UniSAR baseline data hand-off; frozen `Qwen/Qwen3-0.6B` query features at revision `c1899de289a04d12100db370d81485cdf75e47ca` | Confirm processed Kindle data redistribution permission and applicable derived-feature terms. |
| WuKong | MovieLens Latest, `MovielensLatest_x1`; RecZoo/BARS preprocessing used by the WuKong reproduction; source archive SHA-256 `a809509dfce89d77ca0db2985a92033c829688c1a99af98f0ead0e0f79766d27` | The [versioned MovieLens README](https://files.grouplens.org/datasets/movielens/ml-latest-README.html) describes noncommercial research use and redistribution/transformation under its conditions, while the [current GroupLens dataset page](https://grouplens.org/datasets/movielens/) normally disallows public redistribution. Confirm the exact source version, its terms, and the processed package's attribution conditions before public upload. |
| RankMixer | Public metadata identifies Criteo-5M, frequency threshold 10, and the first-5M chronological split | Confirm the precise raw Criteo release/source URL and its redistribution terms; the local metadata does not identify them sufficiently. |
| OneTrans | KuaiSAR v2 processed from `rec_inter.csv`, `src_inter.csv`, `user_features.csv`, and `item_features.csv`; [v2 source record](https://zenodo.org/records/8181109) | The [official KuaiSAR project](https://kuaisar.github.io/) specifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Preserve attribution, noncommercial and share-alike conditions, and the modification notice below. |

KuaiSAR attribution and modification notice: credit the original KuaiSAR authors and dataset described by the [official project](https://kuaisar.github.io/) and [version 2 record](https://zenodo.org/records/8181109). The OneTrans tables are derived from that dataset; preprocessing builds chronological recommendation samples and event histories, and this v2 release additionally masks held-out outcome columns as described above. These derived KuaiSAR artifacts retain the upstream CC BY-NC-SA 4.0 conditions.

The table combines local provenance evidence with the linked primary licensing sources; it is not a legal determination. This folder is prepared for the owner to review and upload, and does not establish publication permission. The maintainer should resolve the listed licensing facts before public publication. No training weights, upstream model source repositories, or separate upstream license grants are bundled in this data upload directory.
