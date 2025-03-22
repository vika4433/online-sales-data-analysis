"""
Statistical Analysis Module for Online Sales Data Analysis Application

This module provides functions for statistical analysis of the data, including:
- Descriptive statistics
- Correlation analysis
- Time series analysis
- Group-based statistics
"""

import pandas as pd
import numpy as np
from scipy import stats

def get_descriptive_stats(df, columns=None):
    """
    Calculate descriptive statistics for the specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    columns : list, optional
        List of column names to analyze. If None, all numeric columns are used.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing descriptive statistics
    """
    if df is None or df.empty:
        return None
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    else:
        # Filter to include only existing numeric columns
        columns = [col for col in columns if col in df.columns and 
                  pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return None
    
    # Calculate descriptive statistics
    stats_df = df[columns].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Add additional statistics
    stats_df.loc['variance'] = df[columns].var()
    stats_df.loc['skewness'] = df[columns].skew()
    stats_df.loc['kurtosis'] = df[columns].kurtosis()
    
    return stats_df

def get_correlation_matrix(df, columns=None, method='pearson'):
    """
    Calculate the correlation matrix for the specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    columns : list, optional
        List of column names to analyze. If None, all numeric columns are used.
    method : str, default='pearson'
        Correlation method ('pearson', 'kendall', or 'spearman')
        
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    if df is None or df.empty:
        return None
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    else:
        # Filter to include only existing numeric columns
        columns = [col for col in columns if col in df.columns and 
                  pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)
    
    return corr_matrix

def analyze_time_series(df, date_column, value_column, freq='M'):
    """
    Perform time series analysis on the specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column to analyze
    freq : str, default='M'
        Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        Resampled time series data
    """
    if df is None or df.empty or date_column not in df.columns or value_column not in df.columns:
        return None
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        try:
            date_series = pd.to_datetime(df[date_column])
        except:
            return None
    else:
        date_series = df[date_column]
    
    # Create a copy of the dataframe with the datetime index
    ts_df = df[[date_column, value_column]].copy()
    ts_df.set_index(date_column, inplace=True)
    
    # Resample the data
    resampled = ts_df.resample(freq).agg({
        value_column: ['sum', 'mean', 'count', 'min', 'max']
    })
    
    # Flatten the column names
    resampled.columns = [f"{value_column}_{agg}" for _, agg in resampled.columns]
    
    return resampled

def group_statistics(df, group_by, metrics, aggregations=None):
    """
    Calculate statistics grouped by specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    group_by : str or list
        Column(s) to group by
    metrics : str or list
        Column(s) to calculate statistics for
    aggregations : dict, optional
        Dictionary mapping metrics to aggregation functions
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with grouped statistics
    """
    if df is None or df.empty:
        return None
    
    # Convert single column names to lists
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # Check if all columns exist in the dataframe
    for col in group_by + metrics:
        if col not in df.columns:
            print(f"Column '{col}' not found in the dataframe")
            return None
    
    # Default aggregations if none provided
    if aggregations is None:
        aggregations = {}
        for metric in metrics:
            if pd.api.types.is_numeric_dtype(df[metric]):
                aggregations[metric] = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std']
            else:
                aggregations[metric] = ['count', 'nunique']
    
    # Calculate grouped statistics
    grouped_stats = df.groupby(group_by).agg(aggregations)
    
    return grouped_stats

def test_significance(df, group_column, value_column, test_type='t-test'):
    """
    Perform statistical significance tests between groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    group_column : str
        Column containing group labels
    value_column : str
        Column containing values to test
    test_type : str, default='t-test'
        Type of test to perform ('t-test', 'anova', 'chi2')
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    if df is None or df.empty or group_column not in df.columns or value_column not in df.columns:
        return None
    
    # Get unique groups
    groups = df[group_column].unique()
    
    if len(groups) < 2:
        return {"error": "Need at least two groups for comparison"}
    
    results = {"test_type": test_type, "groups": list(groups)}
    
    try:
        if test_type == 't-test' and len(groups) == 2:
            # Perform t-test for two groups
            group1 = df[df[group_column] == groups[0]][value_column].dropna()
            group2 = df[df[group_column] == groups[1]][value_column].dropna()
            
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            results.update({
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            })
            
        elif test_type == 'anova':
            # Perform ANOVA for multiple groups
            group_data = [df[df[group_column] == group][value_column].dropna() for group in groups]
            f_stat, p_value = stats.f_oneway(*group_data)
            results.update({
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            })
            
        elif test_type == 'chi2' and pd.api.types.is_categorical_dtype(df[value_column]) or df[value_column].nunique() < 10:
            # Perform Chi-square test for categorical data
            contingency_table = pd.crosstab(df[group_column], df[value_column])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            results.update({
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant": p_value < 0.05
            })
            
        else:
            results["error"] = f"Invalid test type '{test_type}' or incompatible with data"
            
    except Exception as e:
        results["error"] = str(e)
    
    return results
