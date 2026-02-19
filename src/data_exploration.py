from config import Config
import pandas as pd
from collections import OrderedDict
def data_exploration(df):
    # Segregate numerical and categorical columns
    numerical_cols = df.select_dtypes(exclude='object').columns
    categorical_cols = df.select_dtypes(include='object').columns

    numerical_stats = []

    # Numerical stats
    for i in numerical_cols:

        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR

        outlier_flag = "Has Outliers" if ((df[i] < LW) | (df[i] > UW)).any() else "No Outliers"

        num_stats = OrderedDict({
            "Features": i,
            "Maximum": df[i].max(),
            "Minimum": df[i].min(),
            "Mean": df[i].mean(),
            "Median": df[i].median(),
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Skewness": df[i].skew(),
            "Kurtosis": df[i].kurtosis(),
            "Outlier Comment": outlier_flag
        })

        numerical_stats.append(num_stats)

    numerical_stats_report = pd.DataFrame(numerical_stats)

    # Categorical stats
    categorical_stats = []

    for i in categorical_cols:

        cat_stats = OrderedDict({
            "Features": i,
            "Unique_Values": df[i].nunique(),
            "Mode": df[i].mode()[0],
            "Value_Counts": df[i].value_counts().to_dict()
        })

        categorical_stats.append(cat_stats)

    categorical_stats_report = pd.DataFrame(categorical_stats)

    return numerical_stats_report, categorical_stats_report
