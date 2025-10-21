# Machine Learning-Driven Prediction of Ultrafast Spin Relaxation in Metal Halide Perovskites for Spintronic Applications

This repository accompanies the paper:

> **Machine Learning-Driven Prediction of Ultrafast Spin Relaxation in Metal Halide Perovskites for Spintronic Applications**  
> *Mingxi CHEN*, 2025.

A beginner-friendly, reproducible template to predict **Spin relaxation rate** using a **Multi-Layer Perceptron (MLP)** with **Leave-One-Out Cross-Validation (LOOCV)**.  
Everything below is step-by-step. You can copy–paste the commands exactly.

---

## 🟢 0. What you need (once)
- Install **Python 3.10+**
- Open a terminal in this folder (right-click → “Open in Terminal” or open PowerShell/CMD here)

Create a clean environment and install the packages:
```bash
pip install -r requirements.txt
```
*Windows users:* you can also double‑click **run.bat** after you put your data in place (it calls the training script).

---

## 📁 1. Put your files
- Put your dataset at: `data/data.xlsx`
- Put your Word glossary at: `docs/Abbreviation of discriptors.docx`  
  (this file explains every descriptor column; a placeholder is already here)

**Data schema (required columns):**
| Column                | Type  | Notes                                               |
|----------------------|-------|-----------------------------------------------------|
| sample               | str   | Sample identifier                                  |
| Spin relaxation rate | float | Target ≥ 0 (the script clips negatives to 0)       |
| … descriptor columns | float | Numerical descriptors (see the Word glossary)      |

> A tiny example `data.xlsx` is already included so you can test the script end‑to‑end.

---

## ▶️ 2. Train the model (LOOCV, beginner-safe)
From the repository root:
```bash
python src/train_mlp_loocv.py
```
This will:
1. Load `data/data.xlsx`
2. Standardize features (`StandardScaler`)
3. Run LOOCV `GridSearchCV` over:
   - `hidden_layer_sizes`: (20,), (30,), (50,), (100,), (10,10), (40,10)
   - `activation`: relu, tanh
   - `alpha`: 1e-4, 1e-3, 1e-2
4. Save:
   - `models/best_model.pkl` — best MLP
   - `models/scaler.pkl` — the fitted StandardScaler
5. Produce figures and a results table:
   - `figures/iteration_vs_loss.png`
   - `figures/model_performance.png`
   - `figures/residuals_histogram.png`
   - `outputs/predictions_vs_experiments.xlsx`

**Runtime tip:** LOOCV takes one fold per sample. With larger datasets, it can be slow.  
For quick checks, use `--cv kfold` (see next section).

---

## 🔮 3. (Optional) Predict on new data
Prepare a file like `data/new_data.xlsx` with the **same descriptor columns** as `data.xlsx` (no target column). Then:
```bash
python src/predict.py --data data/new_data.xlsx --out outputs/predictions_new.xlsx
```
This loads `models/best_model.pkl` + `models/scaler.pkl`, scales and predicts.

---

## ⚙️ 4. Script options (you can ignore if you like defaults)
Both scripts accept optional arguments.

**Training:**
```bash
python src/train_mlp_loocv.py --data data/data.xlsx --target "Spin relaxation rate" --cv loocv
# or faster dev:
python src/train_mlp_loocv.py --cv kfold --k 5
```

**Prediction:**
```bash
python src/predict.py --data data/new_data.xlsx --out outputs/predictions_new.xlsx
```

---

## 🧪 Reproducibility & time
- Random seed is fixed: `random_state=44`
- Example run time note: LOOCV on ~100 samples may take minutes on a typical laptop CPU
- Development tip: iterate with `--cv kfold --k 5`, then finalize with `--cv loocv`

---

## 🩹 Troubleshooting (FAQ)
- **File not found**: Make sure `data/data.xlsx` exists and column names match exactly.
- **Excel read error**: `pip install openpyxl` (already in `requirements.txt`).
- **Slow**: try `--cv kfold --k 5` to speed up, then switch back to LOOCV for final runs.
- **Descriptor mismatch**: Your new data must have **identical** descriptor columns as the training file.

---

## 📤 What you get after training
- `models/best_model.pkl` (MLP)
- `models/scaler.pkl` (StandardScaler)
- `figures/*.png` (loss, scatter, residuals)
- `outputs/predictions_vs_experiments.xlsx`

---

## 📄 Citation
If you use this repository, please cite:
```
@article{Chen2025SpinRelaxation,
  title   = {Machine Learning-Driven Prediction of Ultrafast Spin Relaxation in Metal Halide Perovskites for Spintronic Applications},
  author  = {Mingxi Chen},
  year    = {2025}
}
```

---

## 📝 License
MIT License. See `LICENSE`.

---

## 🤝 Contribution (optional)
Simple PRs are welcome (typos, clarifications). For major changes please open an issue first.

---

## ℹ️ Files you may want to read
- `src/train_mlp_loocv.py` — main training script (beginner-friendly)
- `src/predict.py` — load model + scaler and run predictions on new data
- `docs/Abbreviation of discriptors.docx` — your descriptor glossary (add your real file)
