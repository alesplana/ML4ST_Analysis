import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

def ohe_encoder(df, categorical_features=None):
    onehot = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    encoded_cat = []
    encoded_names = []

    encoded_features = onehot.fit_transform(df[categorical_features])
    encoded_feature_names = onehot.get_feature_names_out(categorical_features)

    df_encoded = pd.concat([df.drop(columns=categorical_features), pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)], axis=1)

    df_encoded['prev_USD_amount'] = df_encoded['prev_USD_amount'].fillna(0) 
    df_encoded['prev_age_delta'] = df_encoded['prev_age_delta'].fillna(0)

    return df_encoded

def mi_calc(df_encoded, categorical_features):
    non_object = df_encoded.select_dtypes(exclude=['object', 'datetime64[ns]']).columns

    mi_scores = mutual_info_classif(df_encoded[non_object[1:]], df_encoded['Label'], n_jobs=5)

    mi_df = pd.DataFrame({
        'Feature': df_encoded[non_object[1:]].columns,
        'MI_Score': mi_scores
    })

    for ohe_ft in categorical_features:
        filtered = mi_df[mi_df['Feature'].str.startswith(ohe_ft)]
        mean_mi_score = filtered['MI_Score'].mean()

        mi_df = mi_df[~mi_df['Feature'].str.startswith(ohe_ft)]

        new_row = pd.DataFrame({'Feature': [ohe_ft], 'MI_Score': [mean_mi_score]})
        mi_df = pd.concat([mi_df, new_row], ignore_index=True)

    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
    return mi_df

    
def plot_mutual_info(mi_df, df_name):
    # Sort values for better visualization
    mi_df_sorted = mi_df.sort_values('MI_Score', ascending=True)
    
    # Create figure with larger size
    plt.figure(figsize=(12, 8))

    typo_prefix = ('is_crossborder', 'under_threshold')

    colors = ['#6BA0AC' if any([typo in feature for typo in typo_prefix]) else '#B2CAF5' for feature in mi_df_sorted['Feature']]
    # Horizontal bar plot
    plt.barh(mi_df_sorted['Feature'], mi_df_sorted['MI_Score'], color=colors)
    plt.title('Mutual Information Scores')
    plt.xlabel('MI Score')
    plt.ylabel('Features')
    
    # Rotate feature labels if needed
    plt.tight_layout()
    plt.savefig(f'../data/feature_selection/{df_name}_mutual_info.png', dpi=300, transparent=True),
    # plt.show()

def plot_mi_elbow(mi_df, df_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mi_df)), mi_df['MI_Score'], 'bo-')
    plt.xlabel('Number of Features')
    plt.ylabel('MI Score')
    plt.title('Elbow Plot of MI Scores')
    plt.grid(True)
    plt.savefig(f'../data/feature_selection/{df_name}_mi_elbow.png', dpi=300, transparent=True)
    # plt.show()

def detect_collinearity_with_mi(df, threshold=0.9):
    n_features = df.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            mi = mutual_info_score(df.iloc[:,i], df.iloc[:,j])
            mi_matrix[i,j] = mi
            mi_matrix[j,i] = mi
    
    def normalize_mi_by_entropy(mi_matrix):
        """
        Normalize MI matrix by dividing by max entropy
        """
        max_entropy = np.log2(len(mi_matrix))  # If using log2
        normalized_matrix = mi_matrix / max_entropy
        return normalized_matrix

    mi_matrix = normalize_mi_by_entropy(mi_matrix)
    
    # Find highly collinear pairs
    collinear_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            if mi_matrix[i,j] > threshold:
                collinear_pairs.append((df.columns[i], df.columns[j], mi_matrix[i,j]))
                
    return mi_matrix, collinear_pairs

def plot_mi_heatmap(mi_matrix, feature_names=None, df_name=''):
    """
    Plot MI matrix heatmap
    
    Parameters:
    mi_matrix: numpy array of mutual information scores
    feature_names: list of feature names (optional)
    """
    plt.figure(figsize=(30, 20))
    
    # If feature names not provided, use indices
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(mi_matrix))]
    
    sns.heatmap(
        pd.DataFrame(mi_matrix, index=feature_names, columns=feature_names),
        annot=True,  # Show numbers
        cmap='Blues',
        vmin=0,
        vmax=np.max(mi_matrix),  # Or set to 1 if normalized
        fmt='.2f'
    )
    
    plt.title('Mutual Information Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'../data/feature_selection/{df_name}_mi_heatmap.png', dpi=300, transparent=True)
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Mutual Information Analysis')
    parser.add_argument('input_file', type=str, help='The input file')
    parser.add_argument('df_name', type=str, help='The name of the dataframe')

    args = parser.parse_args()

    df_name = args.df_name
    df = pd.read_parquet(args.input_file)

    df = df.replace([np.inf], 1e6)
    df = df.replace([-np.inf], -1e6)
    
    categorical_features = ['std_txn_type', 'std_txn_method', 'prev_std_txn_type', 'prev_std_txn_method']
    df = ohe_encoder(df, categorical_features=categorical_features)
    mi_df = mi_calc(df, categorical_features)
    mi_df.to_csv(f'../data/feature_selection/{df_name}_mi_to_target.csv', index=False)

    non_object = df.select_dtypes(exclude=['object', 'datetime64[ns]']).columns

    plot_mutual_info(mi_df, df_name)
    plot_mi_elbow(mi_df, df_name)

    mi_matrix, collinear_pairs = detect_collinearity_with_mi(df[non_object[1:]], threshold=0.5)
    plot_mi_heatmap(mi_matrix, feature_names=df[non_object[1:]].columns, df_name=df_name)
    pd.DataFrame(collinear_pairs, columns=['feature1', 'feature2', 'mi_score']).to_csv(f'../data/feature_selection/{df_name}_col_pairs.csv', index=False)

if __name__ == '__main__':
    main()