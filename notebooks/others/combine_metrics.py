import pandas as pd
import argparse

def add_graph_acct_ids(df):
    def create_account_id(id_col, account_col):
        if pd.isna(id_col) and pd.isna(account_col):
            return None
        elif pd.isna(id_col):
            return account_col
        elif pd.isna(account_col):
            return id_col
        else:
            return f'{id_col}_{account_col}'

    df['party_account_id'] = df.apply(lambda row: create_account_id(row['party_Id'], row['party_Account']), axis=1)
    df['cparty_account_id'] = df.apply(lambda row: create_account_id(row['cparty_Id'], row['cparty_Account']), axis=1)

    return df

def main(args):
    df1 = pd.read_parquet(args.df1)
    # df2 = pd.read_parquet(args.df2)

    print(df1.columns)
    # print(df2.columns)

    df1_bwn = pd.read_parquet(args.df1_bwn).reset_index()
    df1_deg = pd.read_parquet(args.df1_deg).reset_index()
    # df1_wdg = pd.read_parquet(args.df1_wdg).reset_index()

    df1 = add_graph_acct_ids(df1)
    # df2 = add_graph_acct_ids(df2)

    df1_bwn.columns = ['node_id', 'graph_metric_btw']
    df1_deg.columns = ['node_id', 'graph_metric_deg']
    # df1_wdg.columns = ['node_id', 'graph_metric_wdg']

    df1 = df1.merge(df1_bwn, how='left', left_on='party_Id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_btw': 'party_entity_btw'})
    df1 = df1.merge(df1_deg, how='left', left_on='party_Id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_deg': 'party_entity_deg'})

    df1 = df1.merge(df1_bwn, how='left', left_on='party_account_id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_btw': 'party_account_btw'})
    df1 = df1.merge(df1_deg, how='left', left_on='party_account_id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_deg': 'party_account_deg'})

    df1 = df1.merge(df1_bwn, how='left', left_on='cparty_Id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_btw': 'cparty_l1_btw'})
    df1 = df1.merge(df1_deg, how='left', left_on='cparty_Id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_deg': 'cparty_l1_deg'})

    df1 = df1.merge(df1_bwn, how='left', left_on='cparty_account_id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_btw': 'cparty_l2_btw'})
    df1 = df1.merge(df1_deg, how='left', left_on='cparty_account_id', right_on='node_id',).drop('node_id', axis=1).rename(columns={'graph_metric_deg': 'cparty_l2_deg'})

    # First create cparty_is_cash indicator and set its zeros
    df1['cparty_is_cash'] = df1['cparty_Id'].isna().astype(int)
    cash_columns = ['cparty_l1_btw', 'cparty_l1_deg', 'cparty_l2_btw', 'cparty_l2_deg']
    df1.loc[df1['cparty_is_cash'] == 1, cash_columns] = 0

    # Create standalone indicator and set l2 columns to zero
    df1['cparty_is_standalone'] = (
        (~df1['cparty_Id'].isna()) & 
        (df1['cparty_Account'].isna())
    ).astype(int)
    l2_columns = ['cparty_l2_btw', 'cparty_l2_deg']
    df1.loc[df1['cparty_is_standalone'] == 1, l2_columns] = 0

    # Account indicator remains the same
    df1['cparty_is_account'] = (
        (~df1['cparty_Id'].isna()) & 
        (~df1['cparty_Account'].isna())
    ).astype(int)

    df1 = df1.drop(['party_account_id', 'cparty_account_id'], axis=1)

    df1.to_parquet(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine metrics from different sources.')
    parser.add_argument('--df1', required=True, help='Path to the first dataframe parquet file')
    # parser.add_argument('--df2', required=True, help='Path to the second dataframe parquet file')
    parser.add_argument('--df1_bwn', required=True, help='Path to the betweenness parquet file for df1')
    parser.add_argument('--df1_deg', required=True, help='Path to the degree parquet file for df1')
    # parser.add_argument('--df1_wdg', required=True, help='Path to the weighted degree parquet file for df1')
    parser.add_argument('--output', required=True, help='Path to the output parquet file')

    args = parser.parse_args()
    main(args)