import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    import japanize_matplotlib
except ImportError:
    pass

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

print("--- Starting Preprocessing ---")
if os.path.exists('/kaggle/input/allstate-claims-severity/train.csv'):
    base_path = '/kaggle/input/allstate-claims-severity/'
else:
    base_path = '../input/'

train = pd.read_csv(base_path + 'train.csv')
test = pd.read_csv(base_path + 'test.csv')

cont_features = [col for col in train.columns if 'cont' in col]

train['log_loss'] = np.log1p(train['loss'])
train_X = train.drop(['id', 'loss', 'log_loss'], axis=1)
test_X = test.drop(['id'], axis=1)
train_X['is_train'] = 1
test_X['is_train'] = 0
all_data = pd.concat([train_X, test_X], axis=0)

all_data = pd.get_dummies(all_data)

X_train_all = all_data[all_data['is_train'] == 1].drop(['is_train'], axis=1)
X_test_final = all_data[all_data['is_train'] == 0].drop(['is_train'], axis=1)
y_train_all = train['log_loss']

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
y_val_exp = np.expm1(y_val)

# Identify Categorical Columns
all_columns = X_train.columns
cat_col_names = [c for c in all_columns if c not in cont_features]

print("--- 1. XGB (All) ---")
model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1,
    tree_method='hist', device='cuda', early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
T1 = np.expm1(model.predict(X_test_final))
V1 = np.expm1(model.predict(X_val))

print("--- 2. LGB (All) ---")
model_lgb = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1, device='gpu'
)
model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
T2 = np.expm1(model_lgb.predict(X_test_final))
V2 = np.expm1(model_lgb.predict(X_val))

print("--- 3. NN (All) ---")
scaler = StandardScaler()
X_train_nn = X_train.copy()
X_val_nn = X_val.copy()
X_test_nn = X_test_final.copy()
X_train_nn[cont_features] = scaler.fit_transform(X_train_nn[cont_features])
X_val_nn[cont_features] = scaler.transform(X_val_nn[cont_features])
X_test_nn[cont_features] = scaler.transform(X_test_nn[cont_features])

def create_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'), Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

model_nn = create_model(X_train_nn.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_nn.fit(
    X_train_nn, y_train, validation_data=(X_val_nn, y_val), epochs=50, batch_size=256, callbacks=[early_stopping], verbose=0
)
T3 = np.expm1(model_nn.predict(X_test_nn).flatten())
V3 = np.expm1(model_nn.predict(X_val_nn).flatten())

print("--- 4. NN (Cont Only) ---")
X_train_cont = X_train_nn[cont_features]
X_val_cont = X_val_nn[cont_features]
X_test_cont = X_test_nn[cont_features]

model_nn_cont = create_model(X_train_cont.shape[1])
model_nn_cont.fit(
    X_train_cont, y_train, validation_data=(X_val_cont, y_val), epochs=50, batch_size=256, callbacks=[early_stopping], verbose=0
)
T4 = np.expm1(model_nn_cont.predict(X_test_cont).flatten())
V4 = np.expm1(model_nn_cont.predict(X_val_cont).flatten())

print("--- 5. XGB (Cat Only) ---")
X_train_cat = X_train[cat_col_names]
X_val_cat = X_val[cat_col_names]
X_test_cat = X_test_final[cat_col_names]

model_xgb_cat = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1,
    tree_method='hist', device='cuda', early_stopping_rounds=50
)
model_xgb_cat.fit(X_train_cat, y_train, eval_set=[(X_val_cat, y_val)], verbose=False)
T5 = np.expm1(model_xgb_cat.predict(X_test_cat))
V5 = np.expm1(model_xgb_cat.predict(X_val_cat))


print("--- 5-Model Optimization ---")
def loss_func(weights):
    final_pred = (weights[0]*V1) + (weights[1]*V2) + (weights[2]*V3) + (weights[3]*V4) + (weights[4]*V5)
    return mean_absolute_error(y_val_exp, final_pred)

init_weights = [0.2] * 5
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * 5
res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

print('Optimal Weights:')
print(f'1. XGB (All)    : {res.x[0]:.4f}')
print(f'2. LGB (All)    : {res.x[1]:.4f}')
print(f'3. NN (All)     : {res.x[2]:.4f}')
print(f'4. NN (ContOnly): {res.x[3]:.4f}')
print(f'5. XGB (CatOnly): {res.x[4]:.4f}')

pred_ensemble_final = (res.x[0]*T1) + (res.x[1]*T2) + (res.x[2]*T3) + (res.x[3]*T4) + (res.x[4]*T5)

submission_final = pd.DataFrame({
    'id': test['id'],
    'loss': pred_ensemble_final
})
submission_final.to_csv('submission_expert_5models.csv', index=False)
print("submission_expert_5models.csv created successfully!")
