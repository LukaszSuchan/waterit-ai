import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('dataset.csv')

X = data.drop('WATER REQUIREMENT', axis=1)
y = data['WATER REQUIREMENT'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=7, random_state=1, max_depth=20)
model.fit(X, y)

y_pred_forest = model.predict(X_test)

mae_forest = mean_absolute_error(y_test, y_pred_forest)
print("MAE forest:", mae_forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)
print("MSE forest:", mse_forest)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)