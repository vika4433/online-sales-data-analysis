"""
Data Processor Module for Online Sales Data Analysis Application

This module handles data loading, preprocessing, and transformation operations.
It includes functions for:
- Loading data from CSV files
- Handling missing values
- Detecting and removing outliers
- Converting data types
- Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data by converting data types, handling missing values,
    and performing feature engineering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Convert date to datetime
    try:
        processed_df['Date'] = pd.to_datetime(processed_df['Date'], format='%d/%m/%Y')
        
        # Extract useful date components
        processed_df['Year'] = processed_df['Date'].dt.year
        processed_df['Month'] = processed_df['Date'].dt.month
        processed_df['Day'] = processed_df['Date'].dt.day
        processed_df['DayOfWeek'] = processed_df['Date'].dt.dayofweek
        processed_df['MonthName'] = processed_df['Date'].dt.month_name()
    except Exception as e:
        print(f"Error processing date column: {e}")
    
    # Handle missing values
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Replace missing values with median for numeric columns
        if processed_df[col].isnull().sum() > 0:
            median_val = processed_df[col].median()
            processed_df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    # Handle missing values in categorical columns
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Replace missing values with mode for categorical columns
        if processed_df[col].isnull().sum() > 0:
            mode_val = processed_df[col].mode()[0]
            processed_df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")
    
    return processed_df

def detect_outliers(df, columns, contamination=0.05):
    """
    Detect outliers in the specified columns using Isolation Forest.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    columns : list
        List of column names to check for outliers
    contamination : float, default=0.05
        Expected proportion of outliers in the data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with an additional column 'is_outlier' indicating outliers
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Select only numeric columns from the provided list
    valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not valid_columns:
        print("No valid numeric columns provided for outlier detection")
        result_df['is_outlier'] = False
        return result_df
    
    try:
        # Initialize and fit the Isolation Forest model
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(df[valid_columns])
        
        # Predict outliers (1 for inliers, -1 for outliers)
        outliers = iso_forest.predict(df[valid_columns])
        
        # Add outlier indicator to the dataframe (True for outliers)
        result_df['is_outlier'] = outliers == -1
        
        print(f"Detected {result_df['is_outlier'].sum()} outliers out of {len(result_df)} records")
    except Exception as e:
        print(f"Error in outlier detection: {e}")
        result_df['is_outlier'] = False
    
    return result_df

def remove_outliers(df):
    """
    Remove outliers from the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'is_outlier' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if df is None or df.empty or 'is_outlier' not in df.columns:
        return df
    
    # Filter out outliers
    clean_df = df[~df['is_outlier']].copy()
    
    # Drop the outlier indicator column
    if 'is_outlier' in clean_df.columns:
        clean_df.drop('is_outlier', axis=1, inplace=True)
    
    print(f"Removed {len(df) - len(clean_df)} outliers, {len(clean_df)} records remaining")
    return clean_df

def get_data_info(df):
    """
    Get basic information about the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
        
    Returns:
    --------
    dict
        Dictionary containing data information
    """
    if df is None or df.empty:
        return {}
    
    info = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
    }
    
    return info
