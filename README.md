# EEG-based Early Detection of Alzheimer's Disease (AD) and Frontotemporal Dementia (FTD)

## Overview
This project utilizes machine learning techniques to analyze EEG recordings for early detection of neurodegenerative disorders, specifically Alzheimer's Disease (AD) and Frontotemporal Dementia (FTD). The dataset includes EEG resting state-closed eyes recordings from 88 subjects, divided into three groups: AD, FTD, and healthy controls (CN).

## Dataset
- **Participants**:
  - AD group: 36 subjects
  - FTD group: 23 subjects
  - CN group: 29 subjects

- **Cognitive State**:
  - MMSE (Mini-Mental State Examination) score: AD (17.75 ± 4.5), FTD (22.17 ± 8.22), CN (30).
  - Median disease duration: 25 months (IQR: 24 - 28.5 months).

- **Demographics**:
  - Mean age: AD (66.4 ± 7.9), FTD (63.6 ± 8.2), CN (67.9 ± 5.4).

- **Recordings**:
  - Duration: AD (485.5 minutes), FTD (276.5 minutes), CN (402 minutes).
  - Device: Nihon Kohden EEG 2100 clinical device.
  - Electrode setup: 19 scalp electrodes and 2 reference electrodes (A1, A2).
  - Sampling rate: 500 Hz.

## Preprocessing Pipeline
1. Band-pass filter (0.5-45 Hz) applied.
2. Re-referencing signals to A1-A2.
3. Artifact Subspace Reconstruction (ASR) to remove noise.
4. Independent Component Analysis (ICA) to separate signal components.
5. Automatic rejection of eye and jaw artifacts using ICLabel.

## Methodology
### Feature Engineering
- **Techniques**:
  - Relative Band Power (Delta, Theta, Alpha, Beta, Gamma).
  - Dimensionality reduction using PCA.

### Model Development
- **Algorithms**:
  - Baseline models: Random Forest, SVM.
  - Optimized model: XGBoost with hyperparameter tuning using Optuna.

### Interpretability
- **SHAP Analysis**:
  - Global feature importance: SHAP summary plots.
  - Local predictions: SHAP waterfall plots.

## Results
- **Model Performance**:
  - Accuracy: 88.9% (optimized).
  - Precision: 89.8%.
  - Recall: 88.9%.
  - F1-Score: 88.8%.

- **Feature Insights**:
  - Delta and Theta band power showed significant contributions to distinguishing between AD, FTD, and CN.

## Future Directions
- Expand dataset size to improve generalizability.
- Explore multimodal data (e.g., MRI, clinical data).
- Incorporate advanced algorithms (e.g., deep learning, transformers).

## Credits
This project is based on the dataset provided by:
- Andreas Miltiadous et al. (2024). *A dataset of EEG recordings from Alzheimer's disease, Frontotemporal dementia, and Healthy subjects*. DOI: [10.18112/openneuro.ds004504.v1.0.8](https://doi.org/10.18112/openneuro.ds004504.v1.0.8)

## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run preprocessing: `python scripts/preprocessing.py`.
4. Features with GenAI`python scripts/feature_engineering_genai.py`.
5. Train models: `python scripts/model_development.py`.
6. Evaluate models: `python scripts/model_evaluation.py`.
7. Generate the final report: `python scripts/final_report.py`.

---

This project aims to contribute to the early diagnosis of neurodegenerative disorders, leveraging machine learning to enhance non-invasive diagnostic methods.
    
