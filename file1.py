import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Excel file
df = pd.read_excel("Data Sheet.xlsx")

# Select input and output columns
X = df[['Voltage', 'Travel Speed', 'WFS']]
y = df[['BW', 'BH', 'UTS']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
print("Model Evaluation")
print("-------------------")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

#  Extrapolation: Predict on new input values
new_data = pd.DataFrame({
    'Voltage': [19.0, 20.0],         # Custom voltages
    'Travel Speed': [280, 400],     # Custom travel speeds
    'WFS': [2.5, 4.0]               # Custom wire feed speeds
})

new_predictions = model.predict(new_data)
new_results = pd.DataFrame(new_predictions, columns=['BW', 'BH', 'UTS'])
print(" Predicted Outputs for New Inputs:")
print(new_results)
