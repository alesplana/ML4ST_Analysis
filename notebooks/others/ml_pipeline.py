from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

# Define the preprocessing steps as functions
def replace_inf(df):
    return df.replace([np.inf], 1e6).replace([-np.inf], -1e6)

def ohe_encoder(df, categorical_features=None):
    onehot = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
    encoded_features = onehot.fit_transform(df[categorical_features])
    encoded_feature_names = onehot.get_feature_names_out(categorical_features)
    df_encoded = pd.concat([df.drop(columns=categorical_features), pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)], axis=1)
    df_encoded['prev_USD_amount'] = df_encoded['prev_USD_amount'].fillna(0) 
    df_encoded['prev_age_delta'] = df_encoded['prev_age_delta'].fillna(0)
    return df_encoded

# Create the pipeline
pipeline = Pipeline([
    ('preprocessing', ColumnTransformer([
        ('replace_inf', FunctionTransformer(replace_inf)),
        ('ohe_encoder', FunctionTransformer(ohe_encoder), ['categorical_feature_1', 'categorical_feature_2']),
        ('scaler', StandardScaler())
    ])),
    ('smote', SMOTE()),
    ('model', LogisticRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['target'], test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on new data
new_data = ...  # Replace with your new data
predictions = pipeline.predict(new_data)
