import numpy as np
from sklearn.metrics import mean_squared_error

# Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([52, 55, 61, 70, 82])

# -----------------------------
# Model A: Linear Regression
# -----------------------------
coeff_linear = np.polyfit(x, y, 1)
linear_model = np.poly1d(coeff_linear)

y_pred_linear = linear_model(x)
mse_linear = mean_squared_error(y, y_pred_linear)

pred_linear_6 = linear_model(6)

# -----------------------------
# Model B: Polynomial Regression (Degree 4)
# -----------------------------
coeff_poly = np.polyfit(x, y, 4)
poly_model = np.poly1d(coeff_poly)

y_pred_poly = poly_model(x)
mse_poly = mean_squared_error(y, y_pred_poly)

pred_poly_6 = poly_model(6)

# -----------------------------
# Results
# -----------------------------
print("Linear Model Equation:", linear_model)
print("Polynomial Model Equation:", poly_model)

print("\nPrediction at x = 6")
print("Linear Model:", pred_linear_6)
print("Polynomial Model:", pred_poly_6)

print("\nTraining MSE")
print("Linear Model MSE:", mse_linear)
print("Polynomial Model MSE:", mse_poly)
