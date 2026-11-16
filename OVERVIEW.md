# KappaPlace

Official implementation of **KappaPlace** — learning hyperspherical uncertainty for visual place recognition via prototype-anchored supervision.

## 🌟 Key contributions

In summary, our contributions are as follows:

- We introduce *KappaPlace*, a lightweight single-pass framework that models VPR descriptors as von Mises-Fisher (vMF) distributions supervised by geographic prototypes, yielding closed-form query- and match-level confidences without altering frozen backbone recall.
- We extend prototype-anchored uncertainty learning beyond frozen classification backbones to contrastive VPR (via batch-level positive centroids) and end-to-end joint training (*KappaPlace-JT*).
- Across diverse benchmarks, KappaPlace lowers ECE@K, tracks visual degradation and geometric inliers, and improves failure rejection under challenging conditions.

The paper code is two repositories. Each one contains only the code used for the results in the paper. Commands for each repository are in that repository's `README.md`.

| Repository | Role |
| --- | --- |
| **KappaPlace** | Classification training for the three CosPlace backbones, and evaluation of every model in the paper |
| **Contrastive** | Training only, for the frozen MixVPR and SALAD models |

## KappaPlace

Train CosPlace and evaluate every reported model here.

Classification training covers three backbones:

| Backbone | Flag | Descriptor dimension | Image size |
| --- | --- | --- | --- |
| ResNet-50 | `ResNet50` | 512 | 512 |
| ViT-B/16 | `ViT` | 768 | 224 |
| DINOv2-B/14 | `DINOv2-B` | 768 | 224 |

Evaluation covers those three backbones plus the contrastive checkpoints trained in the other repository: MixVPR-512 and SALAD-544 with the descriptor head.

```bash
git clone --recursive <kappaplace-repository-url>
cd KappaPlace-cvpr
```

## Contrastive

Train the frozen contrastive heads here. The retrieval backbone and aggregator stay frozen. Only the kappa head is trained, on SF-XL place batches of 24 places × 20 images, with batch leave-one-out. The anchor for each image is the normalized mean of the other 19 retrieval descriptors from the same place.

| Model name | Backbone | Head | Descriptor dimension | Image size |
| --- | --- | --- | --- | --- |
| MixVPR-512 | ResNet-50 | Aggregator copy, then `Linear(512, 1)` and Softplus | 512 | 320 |
| SALAD-544 | DINOv2-B/14 | Descriptor head: `Linear(544, 1)` and Softplus on the pre-L2 token and cluster sum | 544 | 322 |

Evaluate those checkpoints in KappaPlace.

```bash
git clone <contrastive-repository-url>
cd contrastive
```

## 📊 Dataset preparation

Point `data_folder` at the parent of the VPR benchmarks. Eval sets can be fetched already laid out with [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader).

Contrastive also reads SF-XL place metadata: `places_root` is the SF-XL root (cities `SF3770`–`SF3781`, files `Dataframes/{city}.csv`) and `sfxl_train_root` is the train image root. Image paths come from the `sfxl_rel_path` column.

| Dataset | Config name | Download |
| --- | --- | --- |
| **SF-XL** | `sf_xl` | [CosPlace / SF-XL](https://github.com/gmberton/CosPlace#training-datasets) |
| **SF-XL small** | `sf_xl_small` | Same as SF-XL, small split |
| **Pittsburgh30k** | `pitts30k` | [NetVLAD Pittsburgh](https://www.di.ens.fr/willow/research/netvlad/) (30k split) |
| **MSLS** | `msls-val` | [Mapillary SLS](https://github.com/FrederikWarburg/mapillary_sls) |
| **AmsterTime** | `amstertime` | [AmsterTime](https://github.com/seyrankhademi/AmsterTime) |
| **Nordland** | `nordland` | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) |
| **SPED** | `spedtest` | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) |
| **SVOX** | `svox` | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) (night, sun, snow, rain, overcast) |

```text
data_folder/
├── SF-XL/processed/          # sf_xl
├── SF-XL/small/              # sf_xl_small
├── pitts30k/images/          # pitts30k
├── msls/                     # msls-val
├── amstertime/
├── nordland/
├── SPEDTEST/                 # spedtest
└── svox/images/              # svox
```

Under each benchmark, splits follow `train/`, `val/`, and `test/`, each with `database/` and `queries/` (or `queries_*` / `query*`). Set `data_folder` in the config to the directory that contains these benchmarks.

## 📖 Citation

```bibtex
@article{anonymous2026kappaplace,
  title={KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision},
  author={Anonymous},
  year={2026}
}
```
