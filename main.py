from src.data_ingestion import data_ingestion
from src.data_exploration import data_exploration
from src.data_preprocessing import split_data, split_train_test, create_preprocessor, apply_preprocessing, apply_smote
def main():
    df = data_ingestion()
    numerical_stats_report, categorical_stats_report = data_exploration(df)
    X, y = split_data(df,target_col = 'Late_delivery_risk')
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    preprocessor = create_preprocessor(X)
    X_train_processed, X_test_processed = apply_preprocessing(preprocessor, X_train, X_test)
    X_resampled, y_resampled = apply_smote(X_train, y_train)


    



if __name__ == "__main__":
    main()
