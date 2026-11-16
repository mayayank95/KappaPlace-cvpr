# KappaPlace

Official implementation of **KappaPlace** — learning hyperspherical uncertainty for visual place recognition via prototype-anchored supervision.

The paper code is two repositories. Each one contains only the code used for the results in the paper.

| Repository | Role |
| --- | --- |
| **KappaPlace** | Classification training (CosPlace) and evaluation of every model in the paper |
| **Contrastive** | Training only, for the contrastive models (frozen MixVPR and SALAD) |

Train a MixVPR or SALAD checkpoint in Contrastive, then evaluate it in KappaPlace.

## 🛠 Setup

Tested with **Python 3.9**, **PyTorch 2.8.0**, **torchvision 0.23.0**, and **CUDA 12.8** (cu128 wheel). Paper results used an NVIDIA driver reporting **CUDA 13.2**. A GPU is strongly recommended.

```bash
git clone --recursive <kappaplace-repository-url>
cd KappaPlace-cvpr

python3.9 -m venv .venv
source .venv/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Contrastive (MixVPR and SALAD head training):

```bash
git clone <contrastive-repository-url>
cd contrastive

python3.9 -m venv .venv
source .venv/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Classification training

In **KappaPlace**. The paper trains three CosPlace backbones. The uncertainty head copies that backbone's aggregation, then `Linear(D, 1)` and Softplus. Joint training updates the backbone and the CosFace prototypes together with the head. Early stopping is two-phase: validation recall, then ECE@1, @5, and @10. The recall checkpoint is `best_model_recall.pth`. Post-training freezes the retrieval model, its class weights, and batch-norm, trains the head at `1e-3`, and early-stops on validation ECE@1. The kept weights are `best_model_ece_recall_01.pth`.

| Backbone | `BACKBONE` | Descriptor | Image size |
| --- | --- | --- | --- |
| ResNet-50 | `ResNet50` | 512 | 512 |
| ViT-B/16 | `ViT` | 768 | 224 |
| DINOv2-B/14 | `DINOv2-B` | 768 | 224 |

```bash
bash scripts/cosplace_train.sh
BACKBONE=ViT bash scripts/cosplace_train.sh
BACKBONE=DINOv2-B bash scripts/cosplace_train.sh
```

Post-training loads a checkpoint saved by this trainer (it holds the CosFace prototypes) and freezes it:

```bash
FREEZE=1 RESUME_MODEL=path/to/retrieval.pth bash scripts/cosplace_train.sh
FREEZE=1 RESUME_MODEL=path/to/retrieval.pth BACKBONE=ViT bash scripts/cosplace_train.sh
FREEZE=1 RESUME_MODEL=path/to/retrieval.pth BACKBONE=DINOv2-B bash scripts/cosplace_train.sh
```

One iteration then exit: `DRY_RUN=1 bash scripts/cosplace_train.sh`.

## Evaluation

In **KappaPlace**. The backbone or contrastive head must match the checkpoint. Every benchmark uses `--use_labels`, including Pittsburgh, MSLS, SVOX, SF-XL, AmsterTime, Nordland, and SPED. Positives come from UTM coordinates in the image path.

```bash
bash scripts/cosplace_eval.sh path/to/resnet50.pth
BACKBONE=ViT bash scripts/cosplace_eval.sh path/to/vit.pth
BACKBONE=DINOv2-B bash scripts/cosplace_eval.sh path/to/dinov2.pth

bash scripts/contrastive_eval.sh mixvpr path/to/mixvpr_kappa.pth
bash scripts/contrastive_eval.sh salad-desc path/to/salad_desc_kappa.pth
```

`DATASETS` selects one benchmark. CosPlace defaults to `pitts30k`. The contrastive configs evaluate every entry unless `DATASETS` is set. Inference overhead, from KappaPlace, needs a CUDA GPU:

```bash
python scratch/overhead_benchmark.py
```

## ⚙️ Configuration

KappaPlace configs: `configs/datasets.json` (CosPlace; the train and eval scripts set ResNet-50, ViT-B/16, or DINOv2-B/14), `configs/datasets_mixvpr.json` (MixVPR-512, 320×320, aggregator head), `configs/datasets_salad_desc.json` (SALAD-544, 322×322, descriptor head). Passing 512 for SALAD is rewritten to the official 8448-d model. JSON defaults (`--config`) merge with command-line overrides (`configs/parser.py`).

Common KappaPlace flags:

- **Run & I/O:** `--data_folder`, `--logs_folder`, `--device`, `--num_workers`, `--dry_run`
- **Model:** `--method`, `--backbone`, `--descriptors_dimension`, `--image_size`, `--resume_model`, `--resume_train`
- **Uncertainty:** `--model_mode`, `--uncertainty_loss`, `--var_head_agg`, `--var_head_linear`, `--variance_activation`, `--uncertainty_lambda`, `--freeze_model`
- **Training:** `--batch_size`, `--epochs_num`, `--lr`, `--losses`, `--early_stop_metric`, `--patience`
- **Eval:** `--datasets_type`, `--recall_values`, `--infer_batch_size`, `--save_plots`, `--only_recalls`, `--skip_baselines`

Run `python train.py --help` or `python eval.py --help` in KappaPlace for the full list.

## 📁 Project layout

| Path | Role |
| --- | --- |
| `train.py` | CosPlace training, CosFace classifiers |
| `eval.py` | Retrieval metrics, ECE, rejection curves, optional W&B figures |
| `configs/` | CosPlace / MixVPR / SALAD dataset configs |
| `models/` | Model factory, uncertainty wrapper, CosPlace, MixVPR, SALAD |
| `data/` | Datasets, path resolution, caches |
| `losses/` | CosFace, Gaussian / vMF uncertainty losses |
| `eval_metrics/` | ECE, rejection, baselines, visualizations |
| `semantic_kappa/` | Query-only image degradation and how κ moves |
| `loftr_inliers/` | LoFTR inlier correlation with cosine, joint κ, and L2·κ |
| `utils/` | Runtime setup, W&B, logging helpers |
| `scripts/` | CosPlace train and eval for three backbones; contrastive eval |
| `scratch/` | Ad-hoc utilities (e.g. `overhead_benchmark.py`) |
| `third_party/` | Git submodules: CosPlace, MixVPR, SALAD, STUN, SUE, VPR-methods-evaluation |

`train.py` and `eval.py` use the adapted copies in `models/` and do not import the `third_party/` checkouts.

## ❤️ Acknowledgements

- [CosPlace](https://github.com/gmberton/CosPlace) — classification-based VPR training and SF-XL.
- [MixVPR](https://github.com/amaralibey/MixVPR) — feature-mixing aggregator (paper runs use the 512-d ResNet-50 model).
- [SALAD](https://github.com/serizba/salad) — DINOv2 optimal-transport aggregator (paper runs use 544-d `dino_salad_512_32`).
- [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) — standardized VPR evaluation.
- [SUE](https://github.com/MubarizZaffar/SUE) and [STUN](https://github.com/ramdrop/stun) — uncertainty baselines compared in the paper.

## 🤝 Contributing

Open an issue for bugs or questions. Pull requests are welcome; for larger changes, please open an issue first to discuss the approach.

## 📖 Citation

```bibtex
@article{anonymous2026kappaplace,
  title={KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision},
  author={Anonymous},
  year={2026}
}
```

Also consider citing [CosPlace](https://github.com/gmberton/CosPlace), [MixVPR](https://github.com/amaralibey/MixVPR), [SALAD](https://github.com/serizba/salad), and any baseline methods you compare against (e.g., SUE, STUN).
