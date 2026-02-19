from config import Config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from flaml import AutoML
import pandas as pd
import numpy as np
def split_data(df,target_col = 'Late_delivery_risk'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    print("Split data completed.")
    return X, y

def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Split train and test data completed.")
    return X_train, X_test, y_train, y_test

def create_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    numeric_pipeline = SimpleImputer(strategy="median")

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    print("Preprocessor Created")
    return preprocessor

def apply_preprocessing(preprocessor, X_train, X_test):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing Applied")
    return X_train_processed, X_test_processed

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("SMOTE Applied")
    print("Class Distribution After SMOTE:", np.bincount(y_resampled))

    return X_resampled, y_resampled


