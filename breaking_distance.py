import os
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np

# collection
working_dir = os.getcwd()
df = pd.DataFrame(pd.read_csv(os.path.join(working_dir, 'breaking_distance.csv')))

# splitting
X = df['speed']
y = df['dist']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model selection
model_lin_cross = LinearRegression()
model_lin_cross.fit(X_train_scaled, y_train)
pred_model_lin_cross = model_lin_cross.predict(X_test_scaled)

# cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores_lin = cross_val_score(model_lin_cross, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=kf)

mse_scores_lin = -scores_lin
rmse_scores_lin = np.sqrt(mse_scores_lin)
r2_scores_lin = cross_val_score(model_lin_cross, X_train_scaled, y_train, scoring='r2', cv=kf)

mean_rmse_in = np.mean(rmse_scores_lin)
std_rmse_lin = np.std(rmse_scores_lin)

print(mse_scores_lin.mean(), mean_rmse_in, r2_scores_lin.mean())


X_test_feature = X_test_scaled[:, 0]
X_range_feature = np.linspace(X_test_feature.min(), X_test_feature.max(), 100).reshape(-1, 1)
y_range_pred = model_lin_cross.predict(X_range_feature)

plt.scatter(X_test_feature, y_test, color='blue', label='Actual values') # actual x predicted
plt.plot(X_range_feature, y_range_pred, color='red', linestyle='--', label='Regression line') # regression line

plt.xlabel('Speed')
plt.ylabel('Breaking Distance')
plt.show()