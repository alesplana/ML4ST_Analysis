from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import time
import argparse
import logging

class CustomFunctionTransformer(FunctionTransformer):
    def get_feature_names_out(self, input_features=None):
        return input_features

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.sigmoid(self.fc2(x))
        return x

class RNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=64, batch_size=512, epochs=20, learning_rate=0.001, random_state=None):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _init_model(self, input_size):
        torch.manual_seed(self.random_state)
        self.model = TorchRNN(input_size, self.hidden_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        # Reshape input if needed (for single timestep)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        input_size = X.shape[2]
        self._init_model(input_size)
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            
            for batch_X, batch_y in progress_bar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss/len(train_loader)})
        
        return self

    def predict_proba(self, X):
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        self.model.eval()
        X = torch.FloatTensor(X)
        test_loader = DataLoader(X, batch_size=self.batch_size)
        probas = []
        
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probas.extend(outputs.cpu().numpy())
        
        probas = np.array(probas)
        return np.column_stack((1 - probas, probas))

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

# Define the preprocessing steps as functions
def replace_inf(df):
    df_encoded = df.replace([np.inf], 1e6).replace([-np.inf], -1e6)
    if 'prev_age_delta' in df_encoded.columns:
        df_encoded['prev_age_delta'] = df_encoded['prev_age_delta'].fillna(0)
    if 'prev_USD_amount' in df_encoded.columns:
        df_encoded['prev_USD_amount'] = df_encoded['prev_USD_amount'].fillna(0)
    return df_encoded

def pipeline_init_(model, numerical_columns, categorical_columns, random_state=None):
    pipeline = Pipeline([
        ('preprocessing', ColumnTransformer([
            ('num_pipeline', Pipeline([
                ('replace_inf', CustomFunctionTransformer(replace_inf)),
                ('scaler', MinMaxScaler())
            ]), numerical_columns),
            
            ('cat_pipeline', Pipeline([
                ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), categorical_columns)
        ])),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', model)
    ])

    return pipeline

def get_features(feature_selection_df, n=20):
    typological_features = [
        'is_crossborder', 
        'under_threshold_14d_count',
        'under_threshold_14d_sum',
        'under_threshold_30d_count',
        'under_threshold_30d_sum',
        'under_threshold_7d_count',
        'under_threshold_7d_sum'
        ]
    def_time_columns = ['txn_time_hr', 'txn_time_mm']
    def_categorical_columns = ['std_txn_type', 'std_txn_method', 'prev_std_txn_type', 'prev_std_txn_method']


    features = feature_selection_df[~feature_selection_df['Feature'].isin(typological_features)].nlargest(n, 'MI_Score').iloc[:,0].tolist() + typological_features

    categorical_cols = list(set(features) & set(def_categorical_columns))
    numerical_cols = list(set(features) - (set(categorical_cols) | set(def_time_columns)))

    return features, categorical_cols, numerical_cols

def calculate_informedness_markedness(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate components
    sensitivity = tp / (tp + fn)  # also called TPR
    specificity = tn / (tn + fp)  # also called TNR
    ppv = tp / (tp + fp)  # positive predictive value
    npv = tn / (tn + fn)  # negative predictive value
    
    # Calculate metrics
    informedness = sensitivity + specificity - 1
    markedness = ppv + npv - 1
    
    return informedness, markedness

def get_train_test(train_file, test_file):
    df_train = pd.read_parquet('../data/split/resplit/ds3_train.parquet').drop(columns=['Time_step', 'Transaction_Id', 'Transaction_Type','party_Id',
       'party_Account', 'party_Country', 'cparty_Id', 'cparty_Account',
       'cparty_Country',])
    df_test = pd.read_parquet('../data/split/resplit/ds3_test.parquet').drop(columns=['Time_step', 'Transaction_Id', 'Transaction_Type','party_Id',
        'party_Account', 'party_Country', 'cparty_Id', 'cparty_Account',
        'cparty_Country',])

    X_train  = df_train.drop(columns=['Label'])
    y_train = df_train['Label']

    X_test  = df_test.drop(columns=['Label'])
    y_test = df_test['Label']

    return X_train, y_train, X_test, y_test

def get_model_metrics(test_id, pipeline, X_test, y_test, train_time, result_file):
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:,1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    informedness, markedness = calculate_informedness_markedness(y_test, y_pred)

    with open(result_file, 'w') as file:
        file.write(f'{test_id} Model Results\n')
        file.write(f'ROC AUC: {roc_auc}\n')
        file.write(f'MCC: {mcc}\n')
        file.write(f'Informedness: {informedness}\n')
        file.write(f'Markedness: {markedness}\n\n')
        
        file.write(classification_report(y_test, y_pred))
        file.write(f'\nTime to train: {train_time}\n')

def get_models(model_selection, random_state=None):
    models = {}
    model_choices = ['LogisticRegression', 'NaiveBayes', 'RandomForest', 'GradientBoosting', 'NeuralNetwork']
    
    if model_selection == 0:
        # Run all models
        models['LogisticRegression'] = LogisticRegression(random_state=random_state, max_iter=1000)
        models['NaiveBayes'] = MultinomialNB()
        models['RandomForest'] = RandomForestClassifier(random_state=random_state)
        models['GradientBoosting'] = GradientBoostingClassifier(random_state=random_state)
        models['NeuralNetwork'] = RNNClassifier(random_state=random_state)
    else:
        # Run single selected model
        if model_selection == 1:
            models['LogisticRegression'] = LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_selection == 2:
            models['NaiveBayes'] = MultinomialNB()
        elif model_selection == 3:
            models['RandomForest'] = RandomForestClassifier(random_state=random_state)
        elif model_selection == 4:
            models['GradientBoosting'] = GradientBoostingClassifier(random_state=random_state)
        elif model_selection == 5:
            models['NeuralNetwork'] = RNNClassifier(random_state=random_state)
         
    return models

def get_feature_importance(pipeline, model_name):
    features = pipeline.named_steps['preprocessing'].get_feature_names_out()
    classifier_model = pipeline.named_steps['classifier']
    if model_name == 'LogisticRegression':
        importance = classifier_model.coef_[0]
    elif model_name == 'NaiveBayes':
        log_probs = classifier_model.feature_log_prob_
        importance = log_probs[0] - log_probs[1]
    elif model_name == 'RandomForest' or model_name == 'GradientBoosting':
        importance = classifier_model.feature_importances_
    elif model_name == 'NeuralNetwork':
        importance = np.zeros(len(features))
        logging.warning('Neural Network feature importance not available')

    return pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('train_file', type=str, help='The training file (parquet)')
    parser.add_argument('test_file', type=str, help='The testing file (parquet)')
    parser.add_argument('feature_selection_file', type=str, help='The feature selection file (csv)')
    parser.add_argument('-m', '--model', type=int, default=0, help='The model to use')
    parser.add_argument('-f', '--features', type=int, default=20, help='The number of features to use, excluding typological features')
    parser.add_argument('-r', '--random_state', type=int, default=42, help='The random state')

    args = parser.parse_args()

    dataset_name = args.train_file.split('/')[-1].split('_')[0]
    
    logging.info('Starting ML Pipeline')
    logging.info(f'Training file: {args.train_file}')
    logging.info(f'Testing file: {args.test_file}')
    logging.info(f'Feature selection file: {args.feature_selection_file}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Number of features: {args.features}')
    logging.info(f'Random state: {args.random_state}')

    logging.info('Loading training and testing data')
    X_train, y_train, X_test, y_test = get_train_test(args.train_file, args.test_file)

    logging.info('Loading and selecting features')
    feat_, cat_cols_, num_cols_ = get_features(pd.read_csv(args.feature_selection_file), n=args.features)

    logging.info('Initializing models')
    models = get_models(args.model, args.random_state)

    for model_name, model in models.items():
        if model is not None:
            logging.info(f'Training model: {model_name}')
            pipeline = pipeline_init_(model, num_cols_, cat_cols_, random_state=args.random_state)

            start = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start
            logging.info(f'Model {model_name} trained in {train_time:.2f} seconds')

            logging.info(f'Evaluating model: {model_name}')
            get_model_metrics(f'{dataset_name}_{model_name}', pipeline, X_test, y_test, train_time, f'../data/results/{dataset_name}_{model_name}.txt')

            logging.info(f'Extracting feature importance for model: {model_name}')
            feature_importance = get_feature_importance(pipeline, model_name)
            feature_importance.to_csv(f'../data/results/{dataset_name}_{model_name}_feature_importance.csv', index=False)
            logging.info(f'Feature importance saved for model: {model_name}')
    
    logging.info('ML Pipeline completed')

if __name__ == "__main__":
    main()





