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
    
    # Filter metrics into numeric and non-numeric
    numeric_metrics = []
    non_numeric_metrics = []
    for metric in metrics:
        if pd.api.types.is_numeric_dtype(df[metric]):
            numeric_metrics.append(metric)
        else:
            non_numeric_metrics.append(metric)
    
    # Create separate aggregation dictionaries for numeric and non-numeric columns
    numeric_aggs = {}
    non_numeric_aggs = {}
    
    # If no custom aggregations provided, use defaults
    if aggregations is None:
        for metric in numeric_metrics:
            numeric_aggs[metric] = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std']
        for metric in non_numeric_metrics:
            non_numeric_aggs[metric] = ['count', 'nunique']
    else:
        # Use provided aggregations but ensure they're appropriate for column types
        for metric, aggs in aggregations.items():
            if metric in numeric_metrics:
                numeric_aggs[metric] = aggs
            elif metric in non_numeric_metrics:
                # Filter out numeric aggregations for non-numeric columns
                safe_aggs = [agg for agg in aggs if agg in ['count', 'nunique']]
                if safe_aggs:
                    non_numeric_aggs[metric] = safe_aggs
                else:
                    non_numeric_aggs[metric] = ['count']
    
    # Process numeric and non-numeric columns separately and then combine results
    results = []
    
    # Process numeric columns
    if numeric_aggs:
        try:
            numeric_stats = df.groupby(group_by).agg(numeric_aggs)
            results.append(numeric_stats)
        except Exception as e:
            print(f"Error processing numeric columns: {e}")
    
    # Process non-numeric columns
    if non_numeric_aggs:
        try:
            non_numeric_stats = df.groupby(group_by).agg(non_numeric_aggs)
            results.append(non_numeric_stats)
        except Exception as e:
            print(f"Error processing non-numeric columns: {e}")
    
    # If we have results, combine them
    if results:
        try:
            # Combine the results (this will align on the index)
            combined_stats = pd.concat(results, axis=1)
            return combined_stats
        except Exception as e:
            print(f"Error combining results: {e}")
            # Return the first result if we can't combine
            return results[0] if results else None
    else:
        # Fallback to a simple count if everything else failed
        try:
            return df.groupby(group_by).size().to_frame('count')
        except:
            return None

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
    if df is None or df.empty:
        return {'error': 'Empty dataframe'}
    
    if group_column not in df.columns:
        return {'error': f"Group column '{group_column}' not found"}
    
    if value_column not in df.columns:
        return {'error': f"Value column '{value_column}' not found"}
    
    # Check if value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {'error': f"Value column '{value_column}' must be numeric for significance testing"}
    
    # Get unique groups
    groups = df[group_column].unique()
    
    # Need at least 2 groups for comparison
    if len(groups) < 2:
        return {'error': f"Need at least 2 groups for comparison, found {len(groups)}"}
    
    # Perform t-test (for 2 groups)
    if test_type == 't-test':
        if len(groups) != 2:
            return {'error': f"t-test requires exactly 2 groups, found {len(groups)}"}
        
        group1_data = df[df[group_column] == groups[0]][value_column].dropna()
        group2_data = df[df[group_column] == groups[1]][value_column].dropna()
        
        if len(group1_data) < 2 or len(group2_data) < 2:
            return {'error': "Insufficient data in groups for t-test"}
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        
        return {
            'groups': groups,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Perform ANOVA (for 3+ groups)
    elif test_type == 'anova':
        group_data = [df[df[group_column] == group][value_column].dropna() for group in groups]
        
        # Check if we have enough data in each group
        if any(len(data) < 2 for data in group_data):
            return {'error': "Insufficient data in one or more groups for ANOVA"}
        
        f_stat, p_value = stats.f_oneway(*group_data)
        
        return {
            'groups': groups,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Unsupported test type
    else:
        return {'error': f"Unsupported test type: {test_type}"}
