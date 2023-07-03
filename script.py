import argparse
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

CATEGORICAL_VARS_WITH_NA_MISSING = ['loan_limit', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization']
CATEGORICAL_VARS = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness',
                    'open_credit', 'business_or_commercial', 'Neg_ammortization', 'interest_only',
                    'lump_sum_payment', 'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
                    'credit_type', 'co-applicant_credit_type', 'age', 'submission_of_application', 'Region',
                    'Security_Type']


def load_data(file_path):
    df = pd.read_csv(file_path, sep=";")
    return df


def train_model(df, model_path):
    # Cargar la lista de características
    features = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness',
                'open_credit', 'business_or_commercial', 'loan_amount', 'Neg_ammortization', 'interest_only',
                'lump_sum_payment', 'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
                'credit_type', 'co-applicant_credit_type', 'age', 'submission_of_application', 'Region',
                'Security_Type']


    X = df[features]
    y = df['Status']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=2023)

    modelo_gb = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)

    # Etapa 1: Imputación de valores faltantes en variables categóricas
    categorical_imputer = CategoricalImputer(imputation_method='missing', variables=CATEGORICAL_VARS_WITH_NA_MISSING)

    # Etapa 2: Codificación de variables categóricas nominales
    rare_label_encoder = RareLabelEncoder(n_categories=1, tol=0.01, variables=CATEGORICAL_VARS)
    categorical_encoder = OrdinalEncoder(encoding_method='ordered', variables=CATEGORICAL_VARS)

    # Etapa 3: Escalado de variables
    scaler = MinMaxScaler()

    # Construcción del pipeline
    pipeline = Pipeline([
        ('categorical_imputer', categorical_imputer),
        ('rare_label_encoder', rare_label_encoder),
        ('categorical_encoder', categorical_encoder),
        ('scaler', scaler),
        ('modelo_gradient_boosting', modelo_gb)
    ])

    # Entrenamiento del pipeline
    start_time = datetime.now()
    pipeline.fit(X_train, y_train)
    end_time = datetime.now()

    # Calcular las métricas de entrenamiento
    train_predictions = pipeline.predict(X_train)
    accuracy = accuracy_score(y_train, train_predictions)
    specificity = confusion_matrix(y_train, train_predictions)[0, 0] / (
            confusion_matrix(y_train, train_predictions)[0, 0] + confusion_matrix(y_train, train_predictions)[0, 1])
    sensitivity = confusion_matrix(y_train, train_predictions)[1, 1] / (
            confusion_matrix(y_train, train_predictions)[1, 0] + confusion_matrix(y_train, train_predictions)[1, 1])
    roc_auc = roc_auc_score(y_train, train_predictions)

    # Escribir los resultados en un archivo de texto
    with open('train_results.txt', 'w') as file:
        file.write(f"Fecha y hora de ejecución: {datetime.now()}\n")
        file.write(f"Tiempo de entrenamiento: {end_time - start_time}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Specificity: {specificity}\n")
        file.write(f"Sensitivity: {sensitivity}\n")
        file.write(f"ROC-AUC: {roc_auc}\n")

    joblib.dump(pipeline, model_path)


def predict(df, model_path, output_path):
    # Cargar el modelo entrenado
    pipeline = joblib.load(model_path)

    # Filtrar los datos según las características deseadas
    features = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness',
                'open_credit', 'business_or_commercial', 'loan_amount', 'Neg_ammortization', 'interest_only',
                'lump_sum_payment', 'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
                'credit_type', 'co-applicant_credit_type', 'age', 'submission_of_application', 'Region',
                'Security_Type']
    filtered_df = df[features]

    predictions = pipeline.predict(filtered_df)

    output_df = pd.DataFrame({'ID': df['ID'], 'Status': predictions})

    output_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loan Default Model')
    parser.add_argument('operation', choices=['train', 'predict'], help='Operation to perform (train or predict)')
    parser.add_argument('--data-file', help='Path to the data file')
    parser.add_argument('--model-file', help='Path to the trained model file')
    parser.add_argument('--output-file', help='Path to the output file')

    args = parser.parse_args()

    if args.operation == 'train':
        # Verificar que se proporcionó el archivo de datos
        if not args.data_file:
            print("Error: Data file path is required for training.")
            exit(1)

        data = load_data(args.data_file)

        # Entrenar el modelo y guardar los resultados
        train_model(data, args.model_file)
        print("Training completed.")

    elif args.operation == 'predict':
        # Verificar que se proporcionaron el archivo de datos, el modelo y el archivo de salida
        if not args.data_file or not args.model_file or not args.output_file:
            print("Error: Data file, model file, and output file paths are required for prediction.")
            exit(1)

        data_pred = load_data(args.data_file)

        predict(data_pred, args.model_file, args.output_file)
        print("Prediction completed.")
