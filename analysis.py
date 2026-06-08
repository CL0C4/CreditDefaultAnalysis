import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

credit_df = pd.read_csv('UCI_Credit_Card.csv')

# Education and Marriage columns have undocumented values (0, 5, 6), so regroup into the 'others' category
credit_df['EDUCATION'] = credit_df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
credit_df['MARRIAGE'] = credit_df['MARRIAGE'].replace(0,3)

# Credit Util and Payment Ratio (for September)
# Used '+ 1' to ensure no divide-by-zero errors 
credit_df['CREDIT_UTIL'] = (credit_df['BILL_AMT1'] / (credit_df['LIMIT_BAL'] + 1))
credit_df['PAYMENT_RATIO'] = (credit_df['PAY_AMT1'] / (credit_df['BILL_AMT1'] + 1))

# Some rows become NaN or infinity
# Dropping them has virtually no effect on the predictive model
credit_df = credit_df.replace([np.inf, -np.inf], np.nan)
credit_df = credit_df.dropna(subset=['CREDIT_UTIL', 'PAYMENT_RATIO'])

# Split data into training and testing sets
x = credit_df.drop(columns=['ID', 'default.payment.next.month'])
y = credit_df['default.payment.next.month']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

# Scale data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_bal)
x_test_scaled = scaler.transform(x_test)

# Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train_scaled, y_train_bal)

# predictions (y_pred was used as a baseline, later changed to y_pred_custom)
y_pred = model.predict(x_test_scaled)
y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]

# custom threshold
custom_threshold = 0.15
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

# Evaluation and writing to results.txt
with open("results.txt", 'a', encoding="utf-8") as file:
    file.write(f"\n{custom_threshold} Threshold:\n")
    file.write("Evaluation:\n")
    file.write(classification_report(y_test, y_pred_custom))

    file.write("\nMetrics:\n")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    file.write(f"ROC-AUC Score: {roc_auc:.4f}\n")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom).ravel()
    recall = tp / (tp + fn)
    file.write(f"Recall (Sensitivity): {recall:.4f}\n")
    file.write(f"False Negatives (Missed Defaults): {fn}\n")
    file.write(f"False Positives (Unnecessary Rejections): {fp}\n\n")

# find risk drivers
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_[0]
})

# sort by abs value
importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])

# top 10 drivers
with open("risk_drivers.txt", 'w', encoding="utf-8") as file: 
    file.write("\n--- Top 10 Credit Risk Drivers ---")
    file.write(str(importance_df.head(10)))