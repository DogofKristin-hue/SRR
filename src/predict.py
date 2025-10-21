import argparse
import pandas as pd
import numpy as np
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to Excel with SAME descriptor columns as training data (no target)")
parser.add_argument("--out", default="../outputs/predictions_new.xlsx", help="Where to save predictions")
args = parser.parse_args()

# Load scaler & model
scaler = joblib.load("../models/scaler.pkl")
model = joblib.load("../models/best_model.pkl")

# Load new data
df = pd.read_excel(args.data)
if "sample" in df.columns:
    X = df.drop(columns=["sample"])  # allow a sample column
else:
    X = df

X_scaled = scaler.transform(X.values)
y_pred = np.maximum(model.predict(X_scaled), 0)

out = df.copy()
out["Predicted Spin relaxation rate"] = y_pred
out.to_excel(args.out, index=False)
print(f"✅ Saved predictions to: {args.out}")
