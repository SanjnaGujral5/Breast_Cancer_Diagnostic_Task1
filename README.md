# Breast Cancer Diagnostic Machine Learning Project

A complete, end-to-end Machine Learning project that predicts whether a breast tumor is **Benign** or **Malignant**, using the classic **Wisconsin Breast Cancer Diagnostic Dataset**. Built entirely in Python with scikit-learn — no web frameworks, no Flask, no React, no HTML/CSS. Pure ML, EDA, and good engineering practice.

---

## ⚠️ Safety Disclaimer (Please Read First)

**This project is for academic and educational machine learning purposes only.**

- It is **NOT** a certified or validated medical device.
- It is **NOT** a replacement for professional medical diagnosis.
- Predictions made by this model are **NOT medical advice**.
- Real breast cancer diagnosis requires qualified radiologists, pathologists, and oncologists, supported by clinical procedures such as biopsy and imaging.

**If you or someone you know has a health concern, please consult a qualified medical professional.** This project must never be used to make real healthcare decisions.

---

## 📌 Project Overview

| | |
|---|---|
| **Goal** | Classify breast tumors as Benign or Malignant |
| **Dataset** | Wisconsin Breast Cancer Diagnostic Dataset (`sklearn.datasets.load_breast_cancer`) |
| **Samples** | 569 |
| **Features** | 30 numeric features |
| **Type** | Binary Classification |
| **Best Model** | Logistic Regression (selected automatically by the pipeline based on Recall → ROC-AUC → F1) |
| **Tech Stack** | Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib |

This is **not a web app**. There is no Flask server, no frontend, no API. Everything runs as a Jupyter Notebook and/or plain Python scripts, in line with the project requirements.

---

## 📁 Project Structure

```
BreastCancerDiagnosticML/
├── Breast_Cancer_Diagnostic.ipynb   # Main notebook: EDA + modeling + evaluation (run this first)
├── train_model.py                   # Standalone script: trains all models, tunes, saves final model
├── predict.py                       # Standalone script: loads saved model and makes a prediction
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── models/
│   ├── breast_cancer_model.pkl      # Final trained model (saved with joblib)
│   ├── scaler.pkl                   # Fitted StandardScaler (needed before prediction)
│   ├── feature_names.pkl            # Ordered list of the 30 feature names
│   └── model_name.pkl               # Name of the final selected model
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── class_distribution.png
│   ├── correlation_heatmap_full.png
│   ├── histograms_key_features.png
│   ├── boxplots_key_features.png
│   └── pairplot_key_features.png
└── data/
    └── dataset_description.txt      # Detailed explanation of the dataset and its features
```

---

## 🛠️ Installation

### 1. Clone / copy the project folder

Make sure you have the full `BreastCancerDiagnosticML/` folder with all its files and sub-folders.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`, `notebook`.

> **Optional — XGBoost:** The project will use XGBoost if it's installed, and will **safely skip it** otherwise (no crash, just a printed message). To include it:
> ```bash
> pip install xgboost
> ```

---

## ▶️ How to Run

### Option A — Run the Jupyter Notebook (recommended, most detailed)

```bash
jupyter notebook Breast_Cancer_Diagnostic.ipynb
```

Then run all cells from top to bottom (`Cell → Run All`, or `Kernel → Restart & Run All`). The notebook:
1. Loads and explores the dataset (EDA)
2. Preprocesses the data
3. Trains 6 (or 7, with XGBoost) classification models
4. Evaluates and compares them
5. Tunes the top 2 models
6. Selects, trains, and saves the final model + scaler
7. Generates and saves all visualizations into `outputs/`

### Option B — Run the standalone training script

```bash
python train_model.py
```

This performs the exact same pipeline as the notebook (EDA prints summaries to console instead of showing plots inline, but all key visualizations are still saved to `outputs/`), and saves the final model + scaler to `models/`.

### Option C — Make a prediction using the saved model

```bash
python predict.py
```

You'll be asked to choose:
1. **Manual entry** — type in values for all 30 features and get a prediction.
2. **Quick example** — runs a demo prediction using a real sample from the dataset, so you can test the pipeline without typing 30 numbers.

> **Note:** `predict.py` requires that `models/breast_cancer_model.pkl` and `models/scaler.pkl` already exist. Run `train_model.py` or the notebook first if they're missing.

---

## 📊 Dataset Explanation

The Wisconsin Breast Cancer Diagnostic Dataset contains measurements computed from digitized images of **fine needle aspirate (FNA)** biopsies of breast masses. For each of 10 base characteristics of the cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), three statistics are provided: the **mean**, the **standard error**, and the **"worst" (largest)** value — giving 30 total numeric features.

**Important label encoding:** in this dataset, `target = 0` means **Malignant** and `target = 1` means **Benign**. This project handles that encoding carefully throughout so results are never mislabeled.

Full details are documented in [`data/dataset_description.txt`](data/dataset_description.txt).

---

## 🤖 Model Explanation

We train and compare the following models:

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Linear | Simple, interpretable, strong baseline for this dataset |
| Decision Tree | Tree-based | Easy to interpret, prone to overfitting |
| Random Forest | Ensemble (bagging) | Combines many trees, reduces overfitting |
| SVM | Margin-based | Works very well on this dataset, especially with a linear kernel |
| KNN | Distance-based | Sensitive to feature scaling |
| Naive Bayes | Probabilistic | Fast, assumes feature independence |
| XGBoost *(optional)* | Ensemble (boosting) | Only used if installed; skipped safely otherwise |

All models are trained on **StandardScaler-scaled** features using an **80/20 train-test split** (stratified to preserve class balance).

### Why scaling matters
Features in this dataset range from tiny decimals (e.g. `mean smoothness` ≈ 0.05–0.16) to large numbers (e.g. `mean area` in the hundreds/thousands). Without scaling, large-magnitude features would dominate distance-based and gradient-based models simply due to their size — not their actual predictive power. `StandardScaler` puts every feature on the same footing (mean = 0, std = 1).

### Why Recall matters most here
In a medical screening context:
- A **false negative** (predicting Benign when it's actually Malignant) is the most dangerous error — a real cancer case could go undetected.
- A **false positive** (predicting Malignant when it's actually Benign) causes unnecessary worry, but is far less dangerous since follow-up tests would normally catch it.

Because of this asymmetry, this project selects its final model by prioritizing **Recall on the Malignant class**, followed by **ROC-AUC**, followed by **F1 Score** — not Accuracy alone.

---

## 📈 Result Summary

On the held-out 20% test set (114 samples), after hyperparameter tuning, the pipeline selected **Logistic Regression** as the final model, with results similar to:

| Metric | Score |
|---|---|
| Accuracy | ~98% |
| Precision (Malignant) | ~97-98% |
| Recall (Malignant) | ~97-98% |
| F1 Score (Malignant) | ~97-98% |
| ROC-AUC | ~0.995 |

> Exact numbers may vary slightly depending on your scikit-learn version and environment, but should be very close to the figures above given the fixed `random_state=42` used throughout.

The most influential features for predicting malignancy were related to **size and shape irregularity** — particularly `worst concave points`, `worst perimeter`, `mean concave points`, `worst radius`, and `worst area`. This aligns with medical intuition: malignant tumors tend to be larger and have more irregular, jagged boundaries.

All key plots (confusion matrix, ROC curve, model comparison, feature importance, EDA charts) are saved in the `outputs/` folder.

---

## 🔮 Future Improvements

- Add SHAP / LIME explainability to better interpret individual predictions
- Try additional ensemble techniques (stacking, voting classifiers)
- Perform more advanced feature selection / dimensionality reduction (PCA)
- Add cross-validation-based learning curves to study overfitting/underfitting
- Package the prediction function as an installable Python module
- Add unit tests for the preprocessing and prediction pipeline
- Experiment with deep learning (simple feed-forward neural network) for comparison
- If extended beyond academic use, conduct a proper clinical validation study with domain experts (would be required before any real-world consideration)

---

## ⚠️ Final Reminder

This project, including its final saved model, is intended **strictly for learning machine learning concepts** such as classification, EDA, preprocessing, model comparison, and hyperparameter tuning. It is **not validated for clinical use** and must never replace professional medical advice, diagnosis, or treatment. Always consult a qualified doctor for real health concerns.
