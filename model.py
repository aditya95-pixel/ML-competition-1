import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
import random
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
col1=test_data["Timestamp"].copy()
def preprocess_timestamp(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfMonth'] = df['Timestamp'].dt.day
    df['Year'] = df['Timestamp'].dt.year
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df.drop(['Hour', 'DayOfWeek', 'Month'], axis=1) 

train_data = preprocess_timestamp(train_data)
test_data = preprocess_timestamp(test_data)

X = train_data.drop(['Water_Consumption', 'Timestamp'], axis=1)
y = train_data['Water_Consumption']
test_X = test_data.drop(['Timestamp'], axis=1)

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
test_X[numerical_cols] = num_imputer.transform(test_X[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
test_X[categorical_cols] = cat_imputer.transform(test_X[categorical_cols])

for col in categorical_cols:
    X[col] = X[col].astype('category')
    test_X[col] = test_X[col].astype('category')


train_data = train_data.sort_values('Timestamp')
split_idx = int(0.8 * len(X))
X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

learning_rates = [0.01, 0.03, 0.05, 0.1]
depths = [4, 6, 8, 10]
iterations_options = [500, 1000, 1500]
l2_leaf_regs = [1, 3, 5, 7]

n_iter = 20  
best_score = -np.inf
best_params = None

for i in range(n_iter):
    lr = random.choice(learning_rates)
    depth = random.choice(depths)
    iters = random.choice(iterations_options)
    l2_leaf = random.choice(l2_leaf_regs)
    
    model = CatBoostRegressor(
        cat_features=categorical_cols.tolist(),
        random_state=42,
        learning_rate=lr,
        depth=depth,
        iterations=iters,
        l2_leaf_reg=l2_leaf,
        early_stopping_rounds=50,
        verbose=0
    )
    
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    
    preds = model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    score = max(0, 100 - np.sqrt(mse))
    
    print(f"Iteration {i+1}: lr={lr}, depth={depth}, iters={iters}, l2_leaf={l2_leaf}, score={score:.2f}")
    
    if score > best_score:
        best_score = score
        best_params = {
            "learning_rate": lr,
            "depth": depth,
            "iterations": iters,
            "l2_leaf_reg": l2_leaf
        }
        
print("\nBest score:", best_score)
print("Best hyperparameters:", best_params)

model = CatBoostRegressor(
    cat_features=categorical_cols.tolist(),
    random_state=42,
    learning_rate=best_params["learning_rate"],
    depth=best_params["depth"],
    iterations=best_params["iterations"],
    l2_leaf_reg=best_params["l2_leaf_reg"],
    early_stopping_rounds=50,
    verbose=100  
)

model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

preds = model.predict(X_valid)
score = max(0, 100 - np.sqrt(mean_squared_error(y_valid, preds)))
print(f"Validation Score: {score:.2f}")

test_preds = model.predict(test_X)
submission = pd.DataFrame({'Timestamp':col1, 'Water_Consumption': test_preds})
submission.to_csv('submission.csv', index=False)