# Boston Housing Price Prediction - ML Ops Assignment 1

## 📌 Objective
Implement and automate a machine learning workflow using classical regression models to predict Boston housing prices. The goal is to:
- Compare model performance (MSE / R²)
- Perform hyperparameter tuning
- Set up CI using GitHub Actions

## 🗂 Branch Structure
- `main`: Contains only `README.md`
- `reg`: Classical ML regression models (min. 3 models)
- `hyper`: Adds hyperparameter tuning to `reg` models

## 🔧 Workflow
- `reg` → classical models → merged into `main`
- `hyper` → with tuning → merged into `main`
- CI pipeline triggered on every push

## 🏗 Directory Structure

HousingRegression/
├── .github/
│ └── workflows/
│ └── ci.yml
├── regression.py
├── utils.py
├── requirements.txt
└── README.md

## 📊 Models Compared
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## 🛠 Hyperparameters Tuned
- Linear Regression: normalize, fit_intercept, copy_X
- Decision Tree: max_depth, min_samples_split, min_samples_leaf
- Random Forest: n_estimators, max_depth, max_features

## 📈 Metrics Used
- Mean Squared Error (MSE)
- R² Score

## 📎 Dataset
Due to deprecation, the Boston dataset is loaded manually from:
[http://lib.stat.cmu.edu/datasets/boston](http://lib.stat.cmu.edu/datasets/boston)

## ⚙ CI Integration
GitHub Actions configured to:
- Install dependencies
- Run model pipelines
- Generate report on push

---

