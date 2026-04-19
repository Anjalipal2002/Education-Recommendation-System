import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

os.makedirs("Models", exist_ok=True)

# Load dataset
data = pd.read_csv("student-scores.csv")

# Target column (last column)
target_col = data.columns[-1]
X = data.drop(columns=[target_col])
y = data[target_col]

# Encode categorical features
feature_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save encoders and feature order
joblib.dump(feature_encoders, "Models/feature_encoders.pkl")
joblib.dump(label_encoder, "Models/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "Models/feature_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# SMALL & SAFE MODEL
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    random_state=42,
    n_jobs=1
)

model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(scaler, "Models/scaler.pkl")
joblib.dump(model, "Models/model.pkl")

print("✅ Training completed successfully")
