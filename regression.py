from utils import load_data, evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = load_data()
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# Train and evaluate
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    evaluate_model(name, y, y_pred)
