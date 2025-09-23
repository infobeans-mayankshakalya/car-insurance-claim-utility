import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("synthetic_repairs.csv")

X = df[["make", "model", "year", "damage", "severity"]]
y = df["cost"]

enc = OneHotEncoder(handle_unknown="ignore")
X_enc = enc.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_enc, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {"objective": "reg:squarederror", "eta": 0.1, "max_depth": 5}
bst = xgb.train(params, dtrain, 100, evals=[(dval, "val")], early_stopping_rounds=10)

preds = bst.predict(dval)
print("MAE:", mean_absolute_error(y_val, preds))

joblib.dump(bst, "cost_model.joblib")
joblib.dump(enc, "encoder.joblib")
print("Saved cost_model.joblib and encoder.joblib")
