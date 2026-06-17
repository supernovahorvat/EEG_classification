# EEG Seizure Classification

Binary seizure detection from EEG signals using scalogram-based CNN and ResNet18 models.

---

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with CUDA support is strongly recommended (training on CPU will be very slow)

---

## Installation

**1. Clone the repository and enter the project folder:**

```bash
git clone <repo-url>
cd EEG_classification
```

**2. Create and activate a virtual environment:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

> For GPU support, replace the `torch` and `torchvision` lines with the CUDA build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```
> Replace `cu128` with your CUDA version (e.g. `cu118`, `cu121`). Check yours with `nvidia-smi`.

**4. Launch Jupyter:**

```bash
jupyter notebook
```

---

## Project structure


EEG_classification/
├── data.csv                        EEG data
├── scalograms/                     Scalogram arrays (generated in step 2)
│   ├── X_train.npy / y_train.npy
│   ├── X_val.npy   / y_val.npy
│   └── X_test.npy  / y_test.npy
├── plots/                          Saved figures
├── best_baseline_cnn.pt            Trained Baseline CNN checkpoint (F1)
├── best_baseline_cnn_recall.pt     Trained Baseline CNN checkpoint (Recall)
├── best_resnet18.pt                Trained ResNet18 checkpoint (F1)
└── best_resnet18_recall.pt         Trained ResNet18 checkpoint (Recall)


---

## Running the notebooks

Run the notebooks **in order**. Each step depends on the output of the previous one.

`01_data_preparation.ipynb` | Loads and explores `data.csv`, splits into train / val / test
`02_scalograms.ipynb` | Converts EEG signals to scalogram images, saves `.npy` files to `scalograms/
`03_baseline_cnn_F1.ipynb` | Trains Baseline CNN with Optuna hyperparameter search, optimised for F1 
`04_baseline_cnn_recall.ipynb` | Same as above, optimised for Recall
`05_resnet18_F1.ipynb` | Trains ResNet18 with Optuna hyperparameter search, optimised for F1
`06_resnet18_Recall.ipynb` | Same as above, optimised for Recall
`07_confusion_matrices.ipynb` | Loads all 4 checkpoints and produces a 2×2 confusion matrix figure

> Notebooks 3–6 each run 30 Optuna trials × 5-fold CV and can take ~1 hour per notebook on an RTX 3090. On CPU, expect significantly longer runtimes.

---

## Expected model run outputs

Baseline CNN (F1) results:
- plots/cnn_loss.png
- plots/cnn_confusion.png
- plots/cnn_roc.png 

Baseline CNN (Recall) results
- plots/cnn_recall_loss.png
- plots/cnn_recall_confusion.png
- plots/cnn_recall_roc.png` 

ResNet18 (F1) results
- plots/resnet18_loss.png
- plots/resnet18_confusion.png
- plots/resnet18_roc.png

ResNet18 (Recall) results

- plots/resnet18_recall_loss.png
- plots/resnet18_recall_confusion.png
- plots/resnet18_recall_roc.png

Combined 2×2 confusion matrix figure
- plots/confusion_matrices_2x2.png 


Developed with assistance from `Claude Code (Anthropic)`