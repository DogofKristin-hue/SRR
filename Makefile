train:
	python src/train_mlp_loocv.py --data data/data.xlsx --target "Spin relaxation rate"

predict:
	python src/predict.py --data data/new_data.xlsx --out outputs/predictions_new.xlsx
