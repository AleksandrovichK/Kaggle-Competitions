import sys
import warnings

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
print("Imports have been set")

# Disabling warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Reading the data to X
X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
# Reading the test data to X_test_full
X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')


# Showing general info
print("Data:")
print(X.head())
# print(X.describe())


# Rows before:
rows_before = X.shape[0]
# Removing rows with missing/empty target
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
# Rows after:
rows_after = X.shape[0]
print("Rows containing NaN in SalePrice were dropped: " + str(rows_before - rows_after))

# Separating target from predictors
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Spltting X and y to train and validation sets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print("Shape of X_train: " + str(X_train_full.shape) + ", shape of y_train: " + str(y_train.shape))
print("Shape of X_valid_full: " + str(X_valid_full.shape) + ", shape of y_valid: " + str(y_valid.shape))

# Select categorical columns
categorical_columns = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]
print("\nColumns whice are categorical (suitable for OneHot encoding): " + str(len(categorical_columns)) + " out of " + str(X_train_full.shape[1]))
print(categorical_columns)

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
print("\nColumns which are numeric (no encoding required): " + str(len(numeric_cols)) + " out of " + str(X_train_full.shape[1]))
print(numeric_cols)

nan_count = (X_train_full.isnull().sum())
nan_count = nan_count[nan_count > 0].sort_values(ascending=False)
print("\nColums containig NaN: ")
print(nan_count)

columns_containig_nan = nan_count.index.to_list()
print("\nWhat values they contain: ")
print(X_train_full[columns_containig_nan])

numeric_cols_nan = [column for column in numeric_cols if column in columns_containig_nan]
categorical_cols_nan = [column for column in categorical_columns if column in columns_containig_nan]

print("\nNumeric with nan are " + str(len(numeric_cols_nan)) + "  : " + str(numeric_cols_nan))
print("Categor with nan are " + str(len(categorical_cols_nan)) + " : " + str(categorical_cols_nan))

# This imputer imputes 0 to numeric values
imputer_numeric = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
])

# This imputer imputs 'missing_value' to categoric values
imputer_categoric = Pipeline(
    steps=[('imputer',
            SimpleImputer(strategy='constant', fill_value='missing_value'))])

preprocessor = ColumnTransformer(transformers=[('imputer_numeric',
                                                imputer_numeric,
                                                numeric_cols),
                                               ('imputer_categoric',
                                                imputer_categoric,
                                                categorical_columns)])

imputed_X_train = pd.DataFrame(preprocessor.fit_transform(X_train_full))
imputed_X_valid = pd.DataFrame(preprocessor.transform(X_valid_full))

# Returning columns back
imputed_X_train.columns = X_train_full.columns
imputed_X_valid.columns = X_valid_full.columns

# print("Before Imputing: ")
# print([elm for elm in X_train_full['YrSold']])

# WELL DONE!!! imputed_X_train now contains repaired from nan X set
print(X_train_full.shape)
print(imputed_X_train.shape)

# Removing all categorical columns with nan
# X_train_without_nan = X_train_full.drop(columns_containig_nan, axis=1)
# X_valid_without_nan = X_valid_full.drop(columns_containig_nan, axis=1)

# Add imputed
# X_train = pd.concat([X_train_without_nan, imputed_X_train], axis=1)
# X_valid = pd.concat([X_valid_without_nan, imputed_X_valid], axis=1)

# X_train.head()

# print("After Imputing: ")
# print([elm for elm in X_train['YrSold']])

# # Apply label encoder
# label_encoder = LabelEncoder()
# for col in good_label_cols:
#     label_X_train[col] = label_encoder.fit_transform(X_train[col])
#     label_X_valid[col] = label_encoder.transform(X_valid[col])

print("Before OH-encoding X_train shape: " + str(X_train_full.shape) + " and X_valid: " + str(X_valid_full.shape))

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_full[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_full[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
# Removing all low_cardinality_cols columns
num_X_train = X_train.drop(low_cardinality_cols, axis=1)
num_X_valid = X_valid.drop(low_cardinality_cols, axis=1)

# Add one-hot encoded columns to numerical features
X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("After OH-encoding X_train shape: " + str(X_train.shape) + " and X_valid: " + str(X_valid.shape))

#
# **3 COLUMNS ARE NOT USED** !!!
#

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

my_model = XGBRegressor(n_estimators=1000, #optimal
                        early_stopping_rounds=1,
                        learning_rate=0.1,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False)
my_model.fit(X_train.append(X_valid), y_train.append(y_valid))
preds_test = my_model.predict(X_test)

mae = mean_absolute_error(my_model.predict(X_valid), y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
print("Submission file is formed")