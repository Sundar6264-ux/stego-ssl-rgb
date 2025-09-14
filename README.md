# 🛰️ stego-ssl-rgb — Self‑Supervised Learning with Steganographic Perturbations

A compact research repo exploring **Barlow Twins** self‑supervised learning on **CIFAR‑10**
with **bit‑plane steganography** used as an *intentional perturbation/augmentation*.
Two variants are provided:
- **Green‑channel bit‑plane** perturbation, and
- **RGB bit‑plane** perturbation.

The pipeline pretrains an encoder using Barlow Twins (two augmented views), then evaluates
representations with a **linear probe** and **fine‑tuning**; utilities generate **t‑SNE** plots
to visualize separability.

---

## 📦 What’s inside

```
stego-ssl-rgb/
├─ src/
│  ├─ originalBarlow/                 # Baseline Barlow‑Twins training + eval
│  │  ├─ Pymain.py                    # SSL pretraining (clean two‑crop views)
│  │  ├─ lp.py                        # Linear probe (logistic head on frozen encoder)
│  │  ├─ ft_p.py, ft_p_light_adamw.py # Full fine‑tune options
│  │  └─ eval_bt.py, tst.py           # Misc eval helpers
│  ├─ bit_plane_stego_GreenChannel/   # Stego on Green channel
│  │  ├─ Pretrain_Stego_Model.py      # SSL pretraining (view2 uses bit‑plane stego)
│  │  ├─ linear_probe_vis.py          # Linear probe + visualization
│  │  └─ Model_Visualize.py           # t‑SNE / feature visualization
│  └─ bitpl_stego_RGB/                # Stego across RGB
│     ├─ bitplane_stego_train.py      # SSL pretraining with RGB bit‑plane
│     ├─ ftsne.py, fvis.py            # TSNE and feature viz
│     └─ st_ev.py                     # Evaluation helpers
├─ results/                           # Example outputs (accuracy curves, TSNE plots, etc.)
├─ requirements.txt                   # Python dependencies
└─ README.md
```

> **Note:** Scripts are intentionally lightweight research drivers. Some flags are hard‑coded in the files;
> adjust values directly or run with `--help` where available.

---

## 🧠 Method (high‑level)

1. **Two‑view SSL** — For each image, create two views:
   - `view1`: standard augmentations (crop, flip, color jitter, etc.).
   - `view2`: same base augs **plus bit‑plane steganography** (flip/embed a chosen bit‑plane).
2. **Barlow Twins** — Encode both views, project with an MLP, and minimize the cross‑correlation
   objective to align features while reducing redundancy.
3. **Evaluation** — Train a **linear probe** on frozen features and optionally **fine‑tune** the full model.
4. **Visualization** — Produce **t‑SNE** scatter plots of embeddings during/after training.

---

## 🛠 Requirements

- Python **3.9+** (3.11 is fine)
- A recent **PyTorch** + **torchvision** (CPU or CUDA)
- Other Python libs: `numpy`, `Pillow`, `scikit-learn`, `matplotlib`, `tqdm`

Install everything with:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> If your platform needs specific PyTorch wheels (CUDA/cuDNN), prefer the official install selector
> and then `pip install -r requirements.txt --no-deps` to keep your chosen torch/torchvision versions.

---

## 🚀 Quickstart

### 1) SSL pretraining (baseline, clean views)
```bash
python src/originalBarlow/Pymain.py
```
- Downloads **CIFAR‑10** via `torchvision` on first run.
- Outputs checkpoints and logs (see prints).

### 2) SSL pretraining with **Green‑channel bit‑plane stego**
```bash
python src/bit_plane_stego_GreenChannel/Pretrain_Stego_Model.py
```

### 3) SSL pretraining with **RGB bit‑plane stego**
```bash
python src/bitpl_stego_RGB/bitplane_stego_train.py
```

### 4) Linear probe / fine‑tune / visualization
```bash
# Linear probe (frozen encoder)
python src/originalBarlow/lp.py

# Full fine‑tune
python src/originalBarlow/ft_p.py               # or ft_p_light_adamw.py

# t‑SNE / feature visualization helpers
python src/bitpl_stego_RGB/ftsne.py
python src/bitpl_stego_RGB/fvis.py
python src/bit_plane_stego_GreenChannel/Model_Visualize.py
```

Artifacts (accuracy curves, TSNE snapshots) will appear under `results/` and alongside the scripts.

---

## 🧪 Notes & Tips

- **Reproducibility:** set seeds inside the scripts if you need strict repeatability.
- **Batch size / LR:** adjust to your hardware. Cosine LR schedulers are common in these scripts.
- **Mixed precision:** some files use `torch.amp` (automatic mixed precision) — works on CUDA and recent CPUs.
- **NaNs:** if loss becomes NaN, reduce LR, clip gradients, or relax augmentations.
- **CIFAR‑10:** images are 32×32; projectors/backbones assume this input size.

---

## 📈 Example outputs (see `results/`)

- `full_ft_accuracy.png`, `full_ft_train_loss.png`, `full_ft_tsne.png` (green‑channel variant)
- `tsne_epoch_*.png` (baseline Barlow Twins)
- `tsne_ssl.png` (RGB stego variant)

---

## 🧾 License & Citation

Use any OSI license you prefer (e.g., MIT). If you publish work using this repo, please cite **Barlow Twins** and (optionally) this repository.

**Barlow Twins (ICML 2021) — preferred**  
Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* In **Proceedings of the 38th International Conference on Machine Learning (ICML)**, PMLR 139, 12310–12320.

**BibTeX (ICML/PMLR)**
```bibtex
@inproceedings{zbontar2021barlow,
  title     = {Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author    = {Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{'e}phane},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {12310--12320},
  year      = {2021},
  publisher = {PMLR}
}
```

**Alternative (arXiv)**  
Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* arXiv:2103.03230.

**BibTeX (arXiv)**
```bibtex
@article{zbontar2021barlow_arxiv,
  title   = {Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author  = {Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{'e}phane},
  journal = {arXiv preprint arXiv:2103.03230},
  year    = {2021}
}
```

**(Optional) Cite this repository**
```bibtex
@misc{mamilla2025stego_ssl_rgb,
  title        = {stego-ssl-rgb: Self-Supervised Learning with Steganographic Perturbations},
  author       = {Mamilla, Soma Sundar},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-username>/stego-ssl-rgb}},
  note         = {Version <tag or commit>},
}
```

## 🙏 Acknowledgments

- PyTorch & TorchVision for models/datasets
- scikit‑learn for t‑SNE
- Community work on self‑supervised learning and steganographic data transforms
