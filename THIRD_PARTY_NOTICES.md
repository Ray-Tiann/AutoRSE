# Source attribution and licensing

AutoRSE's framework, task wrappers, and evaluation service are released under
Apache-2.0. Third-party components retain their own licenses and notices; this
repository's license does not replace them.

| Component | Origin | Notice |
|---|---|---|
| HSTU dense PyTorch reproduction | [Meta generative-recommenders](https://github.com/meta-recsys/generative-recommenders) and HSTU formulation | Apache-2.0 license and attribution retained in the task baseline directory; unused upstream code omitted |
| TIGER core | [phonism/genrec](https://github.com/phonism/genrec) | MIT license retained at `benchmark/tasks/tiger_task/environment/baseline/upstream/LICENSE` |
| WuKong and FuxiCTR | [BANGWu/WuKong](https://github.com/BANGWu/WuKong), [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) | Original MIT and Apache-2.0 license files retained in the baseline directories |
| IntSR implementation | IntSR / DSFNet method and [UniSAR](https://github.com/TengShi-RUC/UniSAR) | Task implementation supplied by the AutoRSE maintainers; provenance below |
| OneTrans PyTorch implementation | [OneTrans](https://github.com/KiraYeetar/OneTrans_Tensorlfow) formulation/reference | Task implementation supplied by the AutoRSE maintainers; provenance below |
| RankMixer PyTorch implementation | [RankMixer paper](https://arxiv.org/abs/2507.15551) | Task implementation supplied by the AutoRSE maintainers |

The upstream UniSAR, DSFNet, and OneTrans reference repositories did not include
a license in the snapshots checked for this release. We do **not** claim those
upstream repositories are Apache-2.0 licensed. The release owner confirmed that
the included task implementations were independently written or authorized for
redistribution. Contact the maintainers about authorship or permission questions.

Public datasets and derived features have separate source terms. The code license
does not relicense Amazon, MovieLens, Criteo, KuaiSAR, or Qwen-derived data. Read
the dataset card and source terms before use or redistribution.
