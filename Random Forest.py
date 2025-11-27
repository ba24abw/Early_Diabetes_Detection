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
from sklearn.svm import SVC
from xgboost import XGBClassifier
from IPython.display import display


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.svm import SVC

file_path = "B:\\Diabetes detection\Healthcare-Diabetes.csv"
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

