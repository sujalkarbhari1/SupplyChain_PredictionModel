import logging
logging.basicConfig(level=logging.INFO,
                    filename='classification_model.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
import pickle
from src.data_ingestion import data_ingestion
from src.data_exploration import data_exploration
from src.data_preprocessing import split_data, split_train_test, create_preprocessor, apply_preprocessing, apply_smote
from src.model_build import train_flaml
from src.model_evaluation import evaluate_model
def main():
    logging.info('Data Loading Started')
    df = data_ingestion()
    logging.info('Data Loaded Successfully')

    logging.info('Data Exploration Started')
    numerical_stats_report, categorical_stats_report = data_exploration(df)
    logging.info('Data Exploration Completed')
    
    logging.info('Split the data Started')
    X, y = split_data(df)
    logging.info('Split the data Completed')

    logging.info('Train Test Split Started')
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    logging.info('Train Test Split Completed')

    logging.info('Creating a Preprocessor')
    preprocessor = create_preprocessor(X_train)
    logging.info('Preprocessor created successfully')
    
    logging.info('Applying Preprocessing')
    X_train_processed, X_test_processed = apply_preprocessing(preprocessor, X_train, X_test)
    logging.info('Preprocessing Completed')
    
    logging.info('Applying Smote')
    X_resampled, y_resampled = apply_smote(X_train_processed, y_train)
    logging.info('Smote Applied Successfully')

    logging.info('Training Model')
    model = train_flaml(X_resampled, y_resampled)
    logging.info('Training Completed')
    
    logging.info('Model Evaluation Started')
    evaluate_model(model, X_test_processed, y_test)
    logging.info('Model Evaluation Completed')

    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

if __name__ == "__main__":
    main()
