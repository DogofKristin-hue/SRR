import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../data/data.xlsx", help="Path to training Excel file")
parser.add_argument("--target", default="Spin relaxation rate", help="Target column name")
parser.add_argument("--cv", choices=["loocv","kfold"], default="loocv", help="Cross-validation strategy")
parser.add_argument("--k", type=int, default=5, help="k for KFold (when --cv kfold)")
args = parser.parse_args()

os.makedirs("../models", exist_ok=True)
os.makedirs("../figures", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

# Load data
data = pd.read_excel(args.data)
X = data.drop(columns=["sample", args.target])
y = np.maximum(data[args.target].values, 0)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model & grid
model = MLPRegressor(max_iter=5000, random_state=44, solver="adam")
param_grid = {
    "hidden_layer_sizes": [(20,), (30,), (50,), (100,), (10,10), (40,10)],
    "activation": ["relu","tanh"],
    "alpha": [1e-4, 1e-3, 1e-2],
}

cv = LeaveOneOut() if args.cv == "loocv" else KFold(n_splits=args.k, shuffle=True, random_state=44)

grid = GridSearchCV(model, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=None)
grid.fit(X_scaled, y)
best_model = grid.best_estimator_

# Save model & scaler
joblib.dump(best_model, "../models/best_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
print("✅ Saved: models/best_model.pkl & models/scaler.pkl")
print("Best Params:", grid.best_params_)

# Loss curve
plt.figure(figsize=(10,6))
plt.plot(best_model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iteration vs Loss")
plt.grid(True)
plt.savefig("../figures/iteration_vs_loss.png", dpi=300)
plt.close()

# Evaluate on full scaled data (LOOCV grid already uses strict CV for selection)
y_pred = np.maximum(best_model.predict(X_scaled), 0)
mse = mean_squared_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")

# Save predictions
out_df = pd.DataFrame({"Experimental Values": y, "Predicted Values": y_pred})
out_df.to_excel("../outputs/predictions_vs_experiments.xlsx", index=False)

# Scatter
plt.figure(figsize=(8,8))
plt.scatter(y, y_pred, alpha=0.6)
ymin, ymax = float(np.min(y)), float(np.max(y))
plt.plot([ymin, ymax], [ymin, ymax], "r--")
plt.xlabel("Experimental Values")
plt.ylabel("Predicted Values")
plt.title("Experimental vs Predicted Values")
plt.text(ymin, ymax, f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}", fontsize=11, va="top")
plt.grid(True)
plt.savefig("../figures/model_performance.png", dpi=300)
plt.close()

# Residuals
plt.figure(figsize=(10,6))
plt.hist(y - y_pred, bins=30, alpha=0.75, label="Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.legend()
plt.grid(True)
plt.savefig("../figures/residuals_histogram.png", dpi=300)
plt.close()

print("✅ Outputs saved to 'models/', 'figures/', and 'outputs/'")
