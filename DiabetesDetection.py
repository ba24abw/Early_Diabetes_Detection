import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from xgboost import XGBClassifier
from IPython.display import display

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.svm import SVC

file_path = "Healthcare-Diabetes.csv"
df = pd.read_csv(file_path)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
display(df.head())
print("\nValue counts (Outcome):\n", df['Outcome'].value_counts())

# Preprocessing
# Drop Id if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Define X and y
target_col = 'Outcome'
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Columns where 0 means missing in Pima-style datasets
cols_zero_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
cols_zero_missing = [c for c in cols_zero_missing if c in X.columns]

# Replace zeros with NaN for these columns
for c in cols_zero_missing:
    X[c] = X[c].replace(0, np.nan)

print("\nMissing counts after zero->NaN replacement:")
print(X.isnull().sum())

# Median imputation
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Feature selection (SelectKBest)
k = min(6, X_scaled.shape[1])  # top 6 features (adjust k as desired)
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)
mask = selector.get_support()
selected_features = X_scaled.columns[mask].tolist()
feature_scores = pd.DataFrame({'feature': X_scaled.columns, 'score': selector.scores_}).sort_values(by='score', ascending=False)
print("\nFeature ranking (top scores):")
display(feature_scores.head(10))
print("\nSelected top-{} features: {}".format(k, selected_features))

X_sel_df = pd.DataFrame(X_selected, columns=selected_features)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1) Correlation matrix (features + target)
corr_df = pd.concat([X_sel_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
corr_matrix = corr_df.corr()

print("Correlation matrix (features + Outcome):\n", corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation matrix — selected features + Outcome")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_sel_df,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)

# Train model
rf.fit(X_train, y_train)

# Predict class labels
y_pred_rf = rf.predict(X_test)

# Predict probabilities for ROC-AUC
y_pred_rf_proba = rf.predict_proba(X_test)[:, 1]

# Evaluation metrics
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)

print("\n--- Random Forest Classifier Performance ---")
print(f"Accuracy Score     : {acc_rf:.4f}")
print(f"Precision Score    : {prec_rf:.4f}")
print(f"Recall Score       : {rec_rf:.4f}")
print(f"F1 Score           : {f1_rf:.4f}")
print(f"ROC-AUC Score      : {roc_auc_rf:.4f}")

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance Plot
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.bar(range(len(selected_features)), importances[indices])
plt.xticks(range(len(selected_features)),
           [selected_features[i] for i in indices],
           rotation=45)
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.show()

# Train SVM model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predictions
svm_pred = svm_model.predict(X_test)

# Confusion Matrix Visualization

cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance
perm = permutation_importance(svm_model, X_test, y_test, n_repeats=20, random_state=42)
importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X_train.columns[indices], rotation=45)
plt.title("SVM — Feature Importances (Permutation)")
plt.tight_layout()
plt.show()

# Evaluation Metrics
svm_acc = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

print("SVM Accuracy:", svm_acc)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1 Score:", svm_f1)
print("SVM ROC-AUC Score:", svm_roc_auc)

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_pred = xgb_model.predict(X_test)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance Plot
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X_train.columns[indices], rotation=45)
plt.title("XGBoost — Feature Importances")
plt.tight_layout()
plt.show()

# Evaluation Metrics
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print("XGBoost Accuracy:", xgb_acc)
print("XGBoost Precision:", xgb_precision)
print("XGBoost Recall:", xgb_recall)
print("XGBoost F1 Score:", xgb_f1)
print("XGBoost ROC-AUC Score:", xgb_roc_auc)

# Metrics comparison for SVM, Random Forest, and XGBoost

metrics_comparison = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'XGBoost'],
    'Accuracy': [svm_acc, acc_rf, xgb_acc],
    'Precision': [svm_precision, prec_rf, xgb_precision],
    'Recall': [svm_recall, rec_rf, xgb_recall],
    'F1 Score': [svm_f1, f1_rf, xgb_f1],
    'ROC-AUC': [svm_roc_auc, roc_auc_rf, xgb_roc_auc]
})

print(metrics_comparison)

# Plotting the comparison of metrics

plt.figure(figsize=(12, 7))
metrics_comparison.set_index('Model').plot(kind='bar', figsize=(12, 7))

plt.title('Comparison of Evaluation Metrics Across Models')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# Hyperparameter Tuning for SVM
from sklearn.model_selection import GridSearchCV

svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=svm_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("Best SVM Parameters:", svm_grid.best_params_)
print("Best SVM CV Score:", svm_grid.best_score_)

# Final SVM model
best_svm = svm_grid.best_estimator_
svm_pred = best_svm.predict(X_test)
svm_pred_proba = best_svm.predict_proba(X_test)[:, 1]

# Evaluation metrics
svm_acc = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_pred_proba)

print("SVM Test Accuracy:", svm_acc)
print("SVM ROC-AUC:", svm_auc)

# Hyperparameter Tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best RF Parameters:", rf_grid.best_params_)
print("Best RF CV Score:", rf_grid.best_score_)

# Final RF model
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_pred_rf_proba = best_rf.predict_proba(X_test)[:, 1]

# Evaluation metrics
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)

print("Random Forest Test Accuracy:", acc_rf)
print("Random Forest ROC-AUC:", roc_auc_rf)

# Hyperparameter Tuning for XGBoost
from xgboost import XGBClassifier

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1]
}

xgb_grid = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    param_grid=xgb_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("Best XGBoost Parameters:", xgb_grid.best_params_)
print("Best XGBoost CV Score:", xgb_grid.best_score_)

# Final XGBoost model
best_xgb = xgb_grid.best_estimator_
xgb_pred = best_xgb.predict(X_test)
xgb_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

# Evaluation metrics
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

print("XGBoost Test Accuracy:", xgb_acc)
print("XGBoost ROC-AUC:", xgb_auc)

# Comparison table for tuned SVM, Random Forest and XGBoost models

tuned_results = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'XGBoost'],
    'Accuracy': [svm_acc, acc_rf, xgb_acc],
    'Precision': [svm_precision, prec_rf, xgb_precision],
    'Recall': [svm_recall, rec_rf, xgb_recall],
    'F1 Score': [svm_f1, f1_rf, xgb_f1],
    'ROC-AUC': [svm_auc, roc_auc_rf, xgb_auc]
})

print("\nTuned Model Comparison:\n")
print(tuned_results)

# Plot: Comparison of tuned models

plt.figure(figsize=(12, 7))
tuned_results.set_index('Model').plot(kind='bar', figsize=(12, 7))

plt.title('Comparison of Evaluation Metrics for Tuned Models')
plt.xlabel('Model')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

cm_svm = confusion_matrix(y_test, svm_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - SVM (Tuned)")
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot(cmap='Greens')
plt.title("Confusion Matrix - Random Forest (Tuned)")
plt.show()

cm_xgb = confusion_matrix(y_test, xgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp.plot(cmap='Purples')
plt.title("Confusion Matrix - XGBoost (Tuned)")
plt.show()

