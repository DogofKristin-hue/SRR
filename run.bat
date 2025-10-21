@echo off
REM Simple helper to train the model using the default paths.
python src\train_mlp_loocv.py --data data\data.xlsx --target "Spin relaxation rate"
pause
