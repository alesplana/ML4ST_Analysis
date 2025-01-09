import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
import os

# data_dir = '../data/jp_morgan'
data_dir = './ms_dissertation_gh/data/jp_morgan'

df = pl.read_parquet(f'{data_dir}/transaction_metrics_final_aml.parquet')

@dataclass
class ColumnConfig:
    """Configuration for column names in the dataset"""
    date_col: str
    transaction_type_col: str
    transaction_method_col: str
    amount_col: str
    party_col: str
    counterparty_col: str
    account_col: str
    label_col: str  # The column indicating fraud/non-fraud
    positive_label: any  # Value indicating legitimate transaction in label column
    
    @classmethod
    def default(cls) -> 'ColumnConfig':
        """Default column configuration"""
        return cls(
            date_col='date',
            transaction_type_col='transaction_type',
            transaction_method_col='transaction_method',
            amount_col='amount',
            party_col='party',
            counterparty_col='cparty',
            account_col='account_identifier',
            label_col='is_good',
            positive_label=True
        )

class FraudEDA:
    """Fraud Detection Exploratory Data Analysis"""
    
    def __init__(self, column_config: Optional[ColumnConfig] = None):
        self.config = column_config or ColumnConfig.default()
    
    def load_and_clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Initial data loading and cleaning using Polars
        """
        # Convert date column to datetime and extract temporal features
        df = df.with_columns([
            pl.col(self.config.date_col).str.to_datetime().alias('date_parsed'),
            pl.col(self.config.date_col).str.to_datetime().dt.hour().alias('hour'),
            pl.col(self.config.date_col).str.to_datetime().dt.weekday().alias('day_of_week'),
            pl.col(self.config.date_col).str.to_datetime().dt.month().alias('month')
        ])
        
        return df
    
    def basic_statistics(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Calculate basic statistics for numerical columns
        """
        print("\n=== Basic Statistics ===")
        
        # Numerical summary for amount
        amount_stats = df.select([
            pl.col(self.config.amount_col).mean().alias('mean'),
            pl.col(self.config.amount_col).median().alias('median'),
            pl.col(self.config.amount_col).std().alias('std'),
            pl.col(self.config.amount_col).min().alias('min'),
            pl.col(self.config.amount_col).max().alias('max'),
            pl.col(self.config.amount_col).quantile(0.25).alias('25%'),
            pl.col(self.config.amount_col).quantile(0.75).alias('75%')
        ])
        print("\nAmount Statistics:")
        print(amount_stats)

    def add_risk_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add fraud detection specific features
        """
        # Calculate amount z-score per transaction type
        df = df.with_columns([
            (pl.col(self.config.amount_col) - pl.col(self.config.amount_col).mean()) / 
            pl.col(self.config.amount_col).std()
                .over(self.config.transaction_type_col)
                .alias('amount_zscore')
        ])
        
        # Calculate transaction velocity (number of transactions in last hour)
        rolling_df = df.select([
            pl.col(self.config.date_col),
            pl.col(self.config.amount_col)
            .rolling(period='1h', index_column=self.config.date_col)
            .count()
            .alias('rolling_1h_count')
        ])
        
        # Join the rolling_df back to the original df
        df = df.join(rolling_df, on=self.config.date_col, how='left')
        
        return df

    def run_full_eda(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
        """
        Run full exploratory data analysis
        """
        df = self.load_and_clean_data(df)
        df = self.add_risk_features(df)
        stats = self.basic_statistics(df)
        
        return df, stats
    
# Example usage
def main():
    # Define custom column names
    custom_columns = ColumnConfig(
        date_col='Time_step',
        transaction_type_col='std_txn_type',
        transaction_method_col='std_txn_method',
        amount_col='USD_amount',
        party_col='party_Id',
        counterparty_col='cparty_Id',
        account_col='party_Account',
        label_col='Label',
        positive_label='GOOD'
    )
    
    # Initialize the EDA class with custom column configuration
    eda = FraudEDA(column_config=custom_columns)
    
    # Load your data    
    # Run the full EDA
    analyzed_df, results = eda.run_full_eda(df)
    return analyzed_df, results

if __name__ == "__main__":
    main()