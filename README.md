# KappaPlace

Official implementation of **KappaPlace** — learning hyperspherical uncertainty for visual place recognition via prototype-anchored supervision.

## 🌟 Key contributions

- **Prototype-anchored supervision.** Stable latent cluster representatives (class prototypes in [CosPlace](https://github.com/gmberton/CosPlace)) serve as ground-truth targets for a probabilistic objective, enabling aleatoric uncertainty learning while preserving retrieval recall and achieving strong ECE@K.
- **Lightweight vMF uncertainty module.** Image descriptors are modeled as von Mises–Fisher variables on the unit hypersphere; a single forward pass estimates concentration κ via numerically stable vMF NLL. Two variants: **KappaPlace-JT** (joint training) and **KappaPlace-PT** (post-training on a frozen backbone).
- **Match-level uncertainty.** A κ-weighted resultant vector over query and reference embeddings yields a principled pairwise reliability score beyond query-centric uncertainty alone.

Across five benchmarks, KappaPlace reduces ECE@K by up to ~50% compared to existing uncertainty methods while maintaining or improving recall.

## ✨ Features

- **Config-first CLI:** JSON defaults (`--config`) merged with command-line overrides (see `configs/parser.py`).
- **Models:** CosPlace, [MixVPR](https://github.com/amaralibey/MixVPR), and [SALAD](https://github.com/serizba/salad), each with an optional uncertainty head. Upstream code is under `third_party/`; the copies used by `train.py` / `eval.py` live in `models/`.
- **Uncertainty modes** (`--model_mode uncertainty`): variance heads (`--var_head_agg`, `--var_head_linear`, `--variance_activation`), optional uncertainty losses, combined with CosFace-style supervision (`--losses`).
- **Evaluation** (`eval.py`): recalls @K, mAP, ECE, rejection curves and deltas, uncertainty statistics, optional W&B logging and plots (`--save_plots`).
- **Optional logging:** [Weights & Biases](https://wandb.ai) via `--use_wandb`.

## 🛠 Setup

This codebase was tested with **Python 3.9**, **PyTorch 2.8.0**, **torchvision 0.23.0**, and **CUDA 12.8** (cu128 wheel). Paper results used an NVIDIA driver reporting **CUDA 13.2**.

```bash
git clone --recursive <repository-url>
cd KappaPlace

python3.9 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# PyTorch first (pick index: cu128 for GPU, cpu for CPU-only — see pytorch.org/get-started/locally/)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

A GPU is strongly recommended for training and full-scale eval. CPU works too, just much slower.

## 📊 Dataset preparation

Download the VPR benchmarks and point `data_folder` in the config at their parent directory. Every benchmark below is an entry in `configs/datasets.json`, `configs/datasets_mixvpr.json`, and `configs/datasets_salad.json`.

Eval sets can be fetched already laid out for this repo with [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader).


| Dataset | Config name | Path under `data_folder` | Role | Download |
| --- | --- | --- | --- | --- |
| **SF-XL** | `sf_xl` | `SF-XL/processed` | Training | [CosPlace / SF-XL](https://github.com/gmberton/CosPlace#training-datasets) |
| **SF-XL small** | `sf_xl_small` | `SF-XL/small` | Sanity check | Same as SF-XL, small split |
| **Pittsburgh30k** | `pitts30k` | `pitts30k/images` | Evaluation | [NetVLAD Pittsburgh](https://www.di.ens.fr/willow/research/netvlad/) (30k split) |
| **MSLS** | `msls-val` | `msls` | Evaluation | [Mapillary SLS](https://github.com/FrederikWarburg/mapillary_sls) |
| **AmsterTime** | `amstertime` | `amstertime` | Evaluation | [AmsterTime](https://github.com/seyrankhademi/AmsterTime) |
| **Nordland** | `nordland` | `nordland` | Evaluation | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) |
| **SPED** | `spedtest` | `SPEDTEST` | Evaluation | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) |
| **SVOX** | `svox` | `svox/images` | Evaluation | [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) (night, sun, snow, rain, overcast) |

### Directory layout

Under each benchmark entry (`name` + relative `path` in the config), splits should follow the layout expected by `data/dataset_paths.py`: `train/`, `val/`, and `test/`, each with `database/` and `queries/` (or `queries_*` / `query*`).

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

1. Edit `configs/datasets.json` (or pass `--config`):
  - Set `data_folder` to the root directory that contains your VPR datasets.
  - Adjust `entries` (`name` + relative `path` per benchmark).

`configs/datasets.json` is included. Set `data_folder` to the directory that contains the benchmarks.

## 🚀 Quick start

Train (joint-training, KappaPlace-JT):

```bash
python train.py --config configs/datasets.json \
  --model_mode uncertainty --losses ce,uncertainty --groups_num 8
```

One training iteration then exit (sanity check):

```bash
python train.py --config configs/datasets.json \
  --model_mode uncertainty --losses ce,uncertainty --groups_num 8 \
  --dry_run --num_workers 0
```

Evaluate a CosPlace checkpoint (`--model_mode` must match training). For UTM-based metrics and `recalls.txt` on benchmarks with pose in paths (e.g. Pittsburgh), add `--use_labels`:

```bash
python eval.py --config configs/datasets.json \
  --resume_model path/to/best_model.pth \
  --model_mode uncertainty \
  --var_head_agg --var_head_linear 1 \
  --datasets pitts30k --datasets_type test --use_labels
```

MixVPR (512-d) and SALAD (544-d, `dino_salad_512_32`) on every benchmark:

```bash
python eval.py --config configs/datasets_mixvpr.json \
  --resume_model path/to/mixvpr.pth

python eval.py --config configs/datasets_salad.json \
  --resume_model path/to/salad.pth
```

Inference overhead (basic vs uncertainty head; **requires a CUDA-capable GPU**):

```bash
python scratch/overhead_benchmark.py
```

## ⚙️ Configuration

- **Configs:** `configs/datasets.json` (CosPlace), `configs/datasets_mixvpr.json` (MixVPR, 512-d, 320×320), `configs/datasets_salad.json` (SALAD, 544-d, 322×322). Passing 512 for SALAD is rewritten to the official 8448-d model.
- **Common CLI flags** (non-exhaustive):
  - **Run & I/O:** `--data_folder`, `--logs_folder`, `--device`, `--num_workers`; `--dry_run`
  - **Model:** `--method`, `--backbone`, `--descriptors_dimension`, `--image_size`, `--resume_model`, `--resume_train`
  - **Uncertainty:** `--model_mode`, `--uncertainty_loss`, `--var_head_agg`, `--var_head_linear`, `--variance_activation`, `--uncertainty_lambda`, `--freeze_model`
  - **Training:** `--batch_size`, `--epochs_num`, `--lr`, `--losses`, `--early_stop_metric`, `--patience`
  - **Eval:** `--datasets_type`, `--recall_values`, `--infer_batch_size`, `--save_plots`, `--only_recalls`, `--skip_baselines`

Run `python train.py --help` or `python eval.py --help` for the full list.

## 📁 Project layout


| Path            | Role                                                                  |
| --------------- | --------------------------------------------------------------------- |
| `train.py`      | Training loop, CosFace classifiers                                    |
| `eval.py`            | Retrieval metrics, ECE, rejection curves, optional W&B figures |
| `configs/`           | `parser.py`, CosPlace / MixVPR / SALAD dataset configs  |
| `models/`            | Model factory, uncertainty wrapper, CosPlace, MixVPR, SALAD |
| `data/`              | Datasets, path resolution, caches                              |
| `losses/`            | CosFace, Gaussian / vMF uncertainty losses                     |
| `eval_metrics/`      | ECE, rejection, baselines, visualizations                      |
| `semantic_kappa/`    | Query-only image degradation and how κ moves                   |
| `loftr_inliers/`     | LoFTR inlier correlation with cosine, joint κ, and L2·κ       |
| `utils/`             | Runtime setup, W&B, logging helpers                            |
| `scratch/`           | Ad-hoc utilities (e.g. `overhead_benchmark.py`)                |
| `third_party/`       | Git submodules: CosPlace, MixVPR, SALAD, STUN, SUE, VPR-methods-evaluation |


## ❤️ Acknowledgements

This repository builds on several open-source VPR projects (under `third_party/` as git submodules). `train.py` and `eval.py` use the adapted copies in `models/` and do not import those checkouts.

- [CosPlace](https://github.com/gmberton/CosPlace) — classification-based VPR training and SF-XL.
- [MixVPR](https://github.com/amaralibey/MixVPR) — feature-mixing aggregator (paper runs use the 512-d ResNet-50 model).
- [SALAD](https://github.com/serizba/salad) — DINOv2 optimal-transport aggregator (paper runs use 544-d `dino_salad_512_32`).
- [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) — standardized VPR evaluation.
- [SUE](https://github.com/MubarizZaffar/SUE) and [STUN](https://github.com/ramdrop/stun) — uncertainty baselines compared in the paper.

## 🤝 Contributing

Open an issue for bugs or questions. Pull requests are welcome; for larger changes, please open an issue first to discuss the approach.

## 📖 Citation

If you use this code in research, please cite:

```bibtex
@article{anonymous2026kappaplace,
  title={KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision},
  author={Anonymous},
  year={2026}
}
```

Also consider citing [CosPlace](https://github.com/gmberton/CosPlace), [MixVPR](https://github.com/amaralibey/MixVPR), [SALAD](https://github.com/serizba/salad), and any baseline methods you compare against (e.g., SUE, STUN).