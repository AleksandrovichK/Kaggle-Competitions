import sys
import warnings

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# TODO
# 1. Add CV
# 2. Check whether is this correct to always imput 0. Maybe it takes sense to impute with 'mean' or another.
# 3. Check performance with LabelEncoder as well. And then check LabelEncoder and OH together.
print("Imports have been set")

# Disabling warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Reading the training/val data and the test data
X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

# Showing general info
print("Data:")
print(X.head())

# Rows before:
rows_before = X.shape[0]
# Removing rows with missing/empty target
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
# Rows after:
rows_after = X.shape[0]
print("\nRows containing NaN in SalePrice were dropped: " + str(rows_before - rows_after))

# Separating target from predictors
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Spltting X and y to train and validation sets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print("\nShape of X_train: " + str(X_train_full.shape) + ", shape of y_train: " + str(y_train.shape))
print("Shape of X_valid_full: " + str(X_valid_full.shape) + ", shape of y_valid: " + str(y_valid.shape))

# Select categorical columns
categoric_columns = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]
print("\nColumns whice are categoric (suitable for OneHot encoding): " + str(len(categoric_columns)) + " out of " + str(X_train_full.shape[1]))
print(categoric_columns)

# Select numeric columns
numeric_columns = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
print("\nColumns which are numeric (no encoding required): " + str(len(numeric_columns)) + " out of " + str(X_train_full.shape[1]))
print(numeric_columns)

nan_count_table = (X_train_full.isnull().sum())
nan_count_table = nan_count_table[nan_count_table > 0].sort_values(ascending=False)
print("\nColums containig NaN: ")
print(nan_count_table)

columns_containig_nan = nan_count_table.index.to_list()
print("\nWhat values they contain: ")
print(X_train_full[columns_containig_nan])

# imputing numeric columns (for a while - all numeric to zero)
for column in numeric_columns:
    X_train_full[column].fillna(value=0, inplace=True)
    X_valid_full[column].fillna(value=0, inplace=True)
    X_test_full[column].fillna(value=0, inplace=True)

# imputng categoric columns
imputer = SimpleImputer(strategy='constant', fill_value='missing_value')
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train_full), columns=X_train_full.columns).astype(X_train_full.dtypes.to_dict())
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid_full), columns=X_valid_full.columns).astype(X_valid_full.dtypes.to_dict())
imputed_X_test = pd.DataFrame(imputer.transform(X_test_full), columns=X_test_full.columns).astype(X_test_full.dtypes.to_dict())

# Sorting in order to observe results
# sorted_before_imput = X_train_full.sort_values('MSSubClass', ascending=True)
# sorted_after_imput = imputed_X_train.sort_values('MSSubClass', ascending=True)

print("\nLet's check whether shape changed or not for X_train. \nBefore imput: " + str(X_train_full.shape) + "\nAfter imput: " + str(imputed_X_train.shape))

nan_count_train_table = (imputed_X_train.isnull().sum())
nan_count_train_table = nan_count_train_table[nan_count_train_table > 0].sort_values(ascending=False)
print("\nAre no NaN here now in train: " + str(nan_count_train_table.size == 0))


nan_count_test_table = (imputed_X_test.isnull().sum())
nan_count_test_table = nan_count_test_table[nan_count_test_table > 0].sort_values(ascending=False)
print("Are no NaN here now in test: " + str(nan_count_test_table.size == 0))
# # Apply label encoder
# label_encoder = LabelEncoder()
# for col in good_label_cols:
#     label_X_train[col] = label_encoder.fit_transform(X_train[col])
#     label_X_valid[col] = label_encoder.transform(X_valid[col])

print("\nBefore OH-encoding X_train shape: " + str(X_train_full.shape) + " and X_valid: " + str(X_valid_full.shape))

# Apply one-hot encoder to each column with categorical data
# NOTE: FOR EACH OF THEM
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train[categoric_columns]), index=X_train_full[categoric_columns].index)
OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_X_valid[categoric_columns]), index=X_valid_full[categoric_columns].index)
OH_cols_test = pd.DataFrame(OH_encoder.transform(imputed_X_test[categoric_columns]), index=X_test_full[categoric_columns].index)

# Remove categorical columns (will replace with one-hot encoding)
# Removing all categoric columns
numeric_X_train = X_train_full.drop(categoric_columns, axis=1)
numeric_X_valid = X_valid_full.drop(categoric_columns, axis=1)
numeric_X_test = X_test_full.drop(categoric_columns, axis=1)

# Add one-hot encoded columns to numerical features
X_train = pd.concat([numeric_X_train, OH_cols_train], axis=1)
X_valid = pd.concat([numeric_X_valid, OH_cols_valid], axis=1)
X_test = pd.concat([numeric_X_test, OH_cols_test], axis=1)

print("After OH-encoding X_train shape: " + str(X_train.shape))
print(", X_valid: " + str(X_valid.shape))
print(", X_test: " + str(X_test.shape))

# Keep selected columns only WHY? HOW ABOUT OTHERS?
# my_cols = low_cardinality_cols + numeric_cols

# Who knows what happens here?
# X_train = X_train_full[my_cols].copy()
# X_valid = X_valid_full[my_cols].copy()
# X_test = X_test_full[my_cols].copy()

# # One-hot encode the data (to shorten the code, we use pandas)
# X_train = pd.get_dummies(X_train)
# X_valid = pd.get_dummies(X_valid)
# X_test = pd.get_dummies(X_test)

# X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
# X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Store for optimal values
# opt_mae = 999999999
# opt_n_estimators = 0
# opt_early_stopping_rounds = 0
# opt_learning_rate = 0

# hpo_array = []

# # Define the model
# for n_estimators in [100, 500, 1000]:
#     for early_stopping_rounds in [1,2,5]:
#         for learning_rate in [0.01, 0.05, 0.1, 0.5, 1, 2]:
#             print("\nn_estimators: " + str(n_estimators))
#             print("early_stopping_rounds: " + str(early_stopping_rounds))
#             print("learning_rate: " + str(learning_rate))
#             my_model_2 = XGBRegressor(random_state=0,
#                                       n_estimators=n_estimators,
#                                       early_stopping_rounds=early_stopping_rounds,
#                                       learning_rate=learning_rate,
#                                       eval_set=[(X_valid, y_valid)],
#                                       verbose=False)
#             # Fit the model
#             my_model_2.fit(X_train, y_train)

#             # Get predictions
#             predictions_2 = my_model_2.predict(X_valid)

#             # Calculate MAE
#             mae_2 = mean_absolute_error(predictions_2, y_valid)

#             hpo_array.append({'mae': mae_2, 'n_estimators':n_estimators, 'early_stopping_rounds':early_stopping_rounds, 'learning_rate': learning_rate})

#             if mae_2 < opt_mae:
#                 opt_mae = mae_2
#                 opt_n_estimators = n_estimators
#                 opt_early_stopping_rounds = early_stopping_rounds
#                 opt_learning_rate = learning_rate

#             # Uncomment to print MAE
#             print("Mean Absolute Error:" , mae_2)

my_model = Pipeline([
    ('scale', StandardScaler()),
    ('reg', XGBRegressor(objective='reg:squarederror',
                         n_estimators=1000,
                         early_stopping_rounds=1,
                         learning_rate=0.1,
                         eval_set=[(X_valid, y_valid)],
                         verbose=False))
])

my_model.fit(X_train, y_train)
preds_test = my_model.predict(X_test)

mae = mean_absolute_error(my_model.predict(X_valid), y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:", mae)

# Save test predictions to file
# output = pd.DataFrame({'Id': X_test.index,
#                        'SalePrice': preds_test})
# output.to_csv('submission.csv', index=False)
# print("Submission file is formed")
