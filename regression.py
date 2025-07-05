from utils import load_data, evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
df = load_data()
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Models and their hyperparameters
model_params = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "positive": [True, False]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "max_features": ['sqrt', 'log2']
        }
    }
}

for name, config in model_params.items():
    print(f"\nRunning GridSearchCV for {name}")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", config["model"])
    ])
    grid = GridSearchCV(pipe, param_grid={
        f"model__{k}": v for k, v in config["params"].items()
    }, cv=5, n_jobs=-1)

    grid.fit(X, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    evaluate_model(name, y, y_pred)
