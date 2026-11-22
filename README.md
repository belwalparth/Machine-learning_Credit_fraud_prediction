# Credit Fraud Prediction

This project uses machine learning to predict whether a transaction is fraudulent from transaction, account, and device features. The analysis is implemented in [`Machine_Learning_Credit_Fraud_analysis.ipynb`](Machine_Learning_Credit_Fraud_analysis.ipynb).

## Notebook Cell Blocks

| Block | Purpose |
|---|---|
| Kaggle setup | Installs Kaggle support, uploads `kaggle.json`, downloads the `playground-series-s6e11` dataset, and extracts `train.csv`. |
| Imports and data inspection | Loads pandas, NumPy, matplotlib, scikit-learn, and the training data; reports shape, fraud counts, and a sample of records. |
| Data split | Keeps only numeric features, separates the `is_fraud` target, and creates fit, validation, and test sets with stratification. |
| Model training | Trains Logistic Regression and Random Forest pipelines. Missing numeric values are median-imputed; Logistic Regression features are also standardized. |
| Cross-validation | Runs three-fold stratified cross-validation on Logistic Regression and compares F1 scores across folds. |
| Threshold selection | Converts model probabilities to predictions at thresholds from 0.05 to 0.95 and selects the model/threshold with the best validation F1. |
| Final evaluation | Retrains the selected Logistic Regression model on the full training portion, evaluates it on the held-out test set, and records precision, recall, F1, average precision, and the confusion matrix. |
| Precision–recall plot | Builds and saves `precision_recall_curve.png` from test-set results. |
| Results export | Saves the final metrics to `results.txt` and downloads the results and plot. |

## Methodology

- **Dataset:** 699,960 transactions with 26 columns.
- **Target:** `is_fraud`; 11,689 fraudulent transactions (1.67%) and 688,271 non-fraudulent transactions (98.33%).
- **Features:** Numeric transaction and account attributes; `id` and `is_fraud` are excluded from model features.
- **Split:** 80% training and 20% held-out test data, with the training portion further split into 80% fit and 20% validation data. All splits preserve the fraud ratio through stratification.
- **Models:** Logistic Regression and Random Forest with 100 trees and maximum depth 8.
- **Preprocessing:** Median imputation for missing values; standardization for Logistic Regression.
- **Validation:** Three-fold stratified cross-validation plus a validation threshold search.
- **Selection criterion:** Highest validation F1, evaluated at thresholds between 0.05 and 0.95 in 0.05 increments.
- **Final test:** Logistic Regression at probability threshold 0.15.

## Results

The held-out test set contains 139,992 transactions: 137,654 non-fraudulent and 2,338 fraudulent.

| Metric | Result |
|---|---:|
| Precision | 0.2999 |
| Recall | 0.3131 |
| F1 | 0.3063 |
| Average precision | 0.2408 |

**Confusion matrix:**

|  | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 135,945 | 1,709 |
| Actual fraud | 1,606 | 732 |

The model correctly identifies 732 of 2,338 fraud cases, but also flags 1,709 legitimate transactions as fraudulent. Overall accuracy is 98%, which is misleading because the dataset is highly imbalanced: precision-recall metrics give a more useful view of fraud detection performance.

## Shortcomings

- **Severe class imbalance:** Fraud represents only 1.67% of observations, limiting sensitivity and making accuracy an unreliable summary metric.
- **Limited feature processing:** Only numeric features are used, so potentially useful categorical fields such as authentication method, channel, and device type are omitted.
- **Basic preprocessing:** Median imputation does not capture feature relationships or missingness patterns.
- **No imbalance handling:** No class weights, resampling, or dedicated anomaly-detection methods are applied.
- **Unoptimized models:** Random Forest depth and Logistic Regression regularization are not tuned extensively.
- **Threshold selection risk:** The threshold is optimized using the validation set and then tested once; repeated or time-based validation could provide a more realistic estimate.
- **No external test or temporal split:** The data is split randomly, so real-world performance may differ if fraud patterns change over time or deployment data differs.
- **Cost not considered:** False positives and false negatives are treated without a business-cost analysis, despite having different operational impacts.
- **Interactive setup:** The notebook requires a Kaggle account, API credentials, and Google Colab uploads to recreate the dataset.

## Run the Notebook

1. Open [`Machine_Learning_Credit_Fraud_analysis.ipynb`](Machine_Learning_Credit_Fraud_analysis.ipynb) in Google Colab or Jupyter.
2. Obtain Kaggle competition credentials and place the downloaded `kaggle.json` in an upload location.
3. Run the cells in order; the first setup cell downloads `playground-series-s6e11` and creates `train.csv`.
4. Review the validation comparison, final test metrics, `results.txt`, and the precision–recall curve.