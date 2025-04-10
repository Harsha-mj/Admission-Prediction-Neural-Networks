import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import logging

logging.basicConfig(level=logging.INFO)

def load_data(path='Admission.csv'):
    data = pd.read_csv(path)
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')
    clean_data = pd.get_dummies(data, columns=['University_Rating', 'Research'], dtype=int)
    return clean_data

def preprocess_and_split(data):
    X = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, X.columns.tolist()  # Return columns

def train_model(xtrain_scaled, ytrain):
    model = MLPClassifier(hidden_layer_sizes=(15, 15), max_iter=300, batch_size=50, random_state=42)
    model.fit(xtrain_scaled, ytrain)
    return model

def evaluate_model(model, xtest_scaled, ytest):
    ypred = model.predict(xtest_scaled)
    accuracy = accuracy_score(ytest, ypred)
    logging.info(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

def save_model(model, scaler, columns, model_filename='model.pkl', scaler_filename='scaler.pkl', columns_filename='columns.pkl'):
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open(columns_filename, 'wb') as columns_file:
        pickle.dump(columns, columns_file)
    logging.info(f"Model, scaler, and column names saved to {model_filename}, {scaler_filename}, and {columns_filename}")

def main():
    data = load_data('Admission.csv')
    xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, columns = preprocess_and_split(data)

    model = train_model(xtrain_scaled, ytrain)
    accuracy = evaluate_model(model, xtest_scaled, ytest)

    if accuracy >= 0.85:
        save_model(model, scaler, columns)
    else:
        logging.warning(f"Accuracy too low: {accuracy:.2f}. Model not saved.")

if __name__ == "__main__":
    main()
