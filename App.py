"""
Interactive Data Analysis Application

This is the main application file that integrates all the modules:
- data_processor.py: For data loading and preprocessing
- statistical_analysis.py: For statistical computations
- visualization.py: For creating interactive visualizations

The application provides an interactive interface for analyzing the Online Sales Data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io

# Import custom modules
import data_processor as dp
import statistical_analysis as sa
import visualization as viz

# Set page configuration
st.set_page_config(
    page_title="Interactive Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main application title
st.markdown('<h1 class="main-header">Interactive Data Analysis Application</h1>', unsafe_allow_html=True)

# Sidebar for data loading and filters
st.sidebar.markdown("## Data Options")

# Data loading section
data_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

@st.cache_data
def load_example_data():
    """Load the example Online Sales Data"""
    try:
        return pd.read_csv("data/Online Sales Data.csv")
    except Exception as e:
        st.error(f"Error loading example data: {e}")
        return None

# Load data
if data_file is not None:
    try:
        df_raw = pd.read_csv(data_file)
        st.sidebar.success(f"Successfully loaded {data_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        df_raw = None
else:
    # Use example data
    if st.sidebar.checkbox("Use example data (Online Sales Data)", value=True):
        df_raw = load_example_data()
        if df_raw is not None:
            st.sidebar.success("Loaded example Online Sales Data")
    else:
        df_raw = None
        st.info("Please upload a CSV file or use the example data")

# If data is loaded, proceed with analysis
if df_raw is not None:
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        df = dp.preprocess_data(df_raw)
    
    # Show data info
    data_info = dp.get_data_info(df)
    
    # Sidebar for data preprocessing options
    st.sidebar.markdown("## Preprocessing Options")
    
    # Outlier detection and removal
    if st.sidebar.checkbox("Detect and Remove Outliers"):
        outlier_columns = st.sidebar.multiselect(
            "Select columns for outlier detection",
            options=df.select_dtypes(include=['float64', 'int64']).columns,
            default=['Units Sold', 'Unit Price', 'Total Revenue']
        )
        
        contamination = st.sidebar.slider(
            "Outlier contamination factor",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Expected proportion of outliers in the data"
        )
        
        if outlier_columns:
            with st.spinner("Detecting outliers..."):
                df_with_outliers = dp.detect_outliers(df, outlier_columns, contamination)
                
                # Show outlier statistics
                n_outliers = df_with_outliers['is_outlier'].sum()
                st.sidebar.info(f"Detected {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
                
                # Option to remove outliers
                if st.sidebar.checkbox("Remove detected outliers"):
                    df = dp.remove_outliers(df_with_outliers)
                else:
                    df = df_with_outliers.copy()
                    if 'is_outlier' in df.columns:
                        df.drop('is_outlier', axis=1, inplace=True)
    
    # Sidebar for filters
    st.sidebar.markdown("## Filters")
    
    # Date filter (if Date column exists and is datetime)
    if 'Date' in df.columns and pd.api.types.is_datetime64_dtype(df['Date']):
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    
    # Categorical filters
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Only add filter for columns with reasonable number of categories
            all_options = ['All'] + sorted(df[col].unique().tolist())
            selected_option = st.sidebar.selectbox(f"Filter by {col}", all_options)
            
            if selected_option != 'All':
                df = df[df[col] == selected_option]
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Explorer", 
        "Statistical Analysis", 
        "Visualizations",
        "Time Series Analysis",
        "Insights"
    ])
    
    # Tab 1: Data Explorer
    with tab1:
        st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
        
        # Data overview
        st.markdown("### Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Records", f"{len(df):,}")
        
        with col2:
            st.metric("Number of Columns", f"{len(df.columns):,}")
        
        with col3:
            st.metric("Memory Usage", f"{data_info['memory_usage']:.2f} MB")
        
        # Show data table with pagination
        st.markdown("### Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Column information
        st.markdown("### Column Information")
        
        # Create a dataframe with column info
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(df[col].dtype) for col in df.columns],  # Convert dtype to string
            'Non-Null Count': [df[col].count() for col in df.columns],
            'Null Count': [df[col].isnull().sum() for col in df.columns],
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Data download option
        st.markdown("### Download Processed Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )
    
    # Tab 2: Statistical Analysis
    with tab2:
        st.markdown('<h2 class="sub-header">Statistical Analysis</h2>', unsafe_allow_html=True)
        
        # Select columns for analysis
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_cols = st.multiselect(
            "Select columns for statistical analysis",
            options=numeric_cols,
            default=list(numeric_cols[:3])  # Default to first 3 numeric columns
        )
        
        if selected_cols:
            # Descriptive statistics
            st.markdown("### Descriptive Statistics")
            stats_df = sa.get_descriptive_stats(df, selected_cols)
            st.dataframe(stats_df, use_container_width=True)
            
            # Correlation analysis
            st.markdown("### Correlation Analysis")
            
            # Select correlation method
            corr_method = st.radio(
                "Correlation method",
                options=["pearson", "spearman", "kendall"],
                horizontal=True
            )
            
            corr_matrix = sa.get_correlation_matrix(df, selected_cols, corr_method)
            
            # Create correlation heatmap
            corr_fig = viz.create_correlation_heatmap(
                df, 
                selected_cols, 
                title=f"{corr_method.capitalize()} Correlation Matrix"
            )
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Group-based statistics
            st.markdown("### Group-based Statistics")
            
            # Select grouping column
            group_cols = st.multiselect(
                "Select columns to group by",
                options=df.select_dtypes(include=['object']).columns,
                default=list(df.select_dtypes(include=['object']).columns[:1]) if len(df.select_dtypes(include=['object']).columns) > 0 else []
            )
            
            if group_cols:
                # Select metrics for grouping - show all columns as options but with info
                all_cols = list(df.columns)
                metric_cols = st.multiselect(
                    "Select metrics to aggregate",
                    options=all_cols,
                    default=list(numeric_cols[:2]) if len(numeric_cols) > 1 else list(numeric_cols[:1])
                )
                
                if metric_cols:
                    # Separate numeric and non-numeric columns
                    selected_numeric = [col for col in metric_cols if pd.api.types.is_numeric_dtype(df[col])]
                    selected_non_numeric = [col for col in metric_cols if not pd.api.types.is_numeric_dtype(df[col])]
                    
                    # Show info about aggregations
                    if selected_numeric:
                        st.info(f"Numeric columns ({', '.join(selected_numeric)}) will be aggregated with count, sum, mean, median, min, max, and std")
                    
                    if selected_non_numeric:
                        st.info(f"Non-numeric columns ({', '.join(selected_non_numeric)}) will only be aggregated with count and unique count")
                    
                    # Get group statistics
                    group_stats = sa.group_statistics(df, group_cols, metric_cols)
                    
                    if group_stats is not None:
                        # Convert the MultiIndex columns to strings to avoid display issues
                        if isinstance(group_stats.columns, pd.MultiIndex):
                            # Create new column names by joining the levels with underscore
                            new_cols = []
                            for col in group_stats.columns:
                                if isinstance(col, tuple):
                                    # Convert any non-string elements to strings
                                    col_elements = [str(x) for x in col]
                                    new_cols.append('_'.join(col_elements))
                                else:
                                    new_cols.append(str(col))
                            
                            # Create a new DataFrame with the string column names
                            flat_stats = pd.DataFrame(
                                group_stats.values, 
                                index=group_stats.index, 
                                columns=new_cols
                            )
                            
                            # Display the flattened DataFrame
                            st.dataframe(flat_stats, use_container_width=True)
                        else:
                            # If it's not a MultiIndex, display as is
                            st.dataframe(group_stats, use_container_width=True)
                    else:
                        st.error("Unable to compute group statistics. Please check your column selections.")
                    
                    # Statistical significance tests
                    if len(group_cols) == 1 and df[group_cols[0]].nunique() >= 2:
                        st.markdown("### Statistical Significance Tests")
                        
                        test_col = st.selectbox(
                            "Select column for significance testing",
                            options=metric_cols
                        )
                        
                        test_type = st.radio(
                            "Test type",
                            options=["t-test", "anova"],
                            horizontal=True,
                            disabled=df[group_cols[0]].nunique() < 2 or df[group_cols[0]].nunique() > 10
                        )
                        
                        if test_col and test_type:
                            test_results = sa.test_significance(df, group_cols[0], test_col, test_type)
                            
                            if test_results and 'error' not in test_results:
                                st.write(f"**Test**: {test_type}")
                                st.write(f"**Groups**: {', '.join(str(g) for g in test_results['groups'])}")
                                
                                if test_type == 't-test':
                                    st.write(f"**t-statistic**: {test_results['t_statistic']:.4f}")
                                elif test_type == 'anova':
                                    st.write(f"**F-statistic**: {test_results['f_statistic']:.4f}")
                                
                                st.write(f"**p-value**: {test_results['p_value']:.4f}")
                                st.write(f"**Significant at α=0.05**: {'Yes' if test_results['significant'] else 'No'}")
                            elif test_results and 'error' in test_results:
                                st.error(test_results['error'])
    
    # Tab 3: Visualizations
    with tab3:
        st.markdown('<h2 class="sub-header">Visualizations</h2>', unsafe_allow_html=True)
        
        # Select chart type
        chart_type = st.selectbox(
            "Select chart type",
            options=["Bar Chart", "Histogram", "Scatter Plot", "Pie Chart", "Box Plot"]
        )
        
        if chart_type == "Bar Chart":
            st.markdown("### Bar Chart")
            
            # Select columns for bar chart
            x_col = st.selectbox("Select X-axis column", options=df.columns)
            y_col = st.selectbox("Select Y-axis column", options=df.select_dtypes(include=['float64', 'int64']).columns)
            color_col = st.selectbox("Select color column (optional)", options=['None'] + list(df.columns))
            orientation = st.radio("Orientation", options=["Vertical", "Horizontal"], horizontal=True)
            
            if x_col and y_col:
                # Create bar chart
                bar_fig = viz.create_bar_chart(
                    df, 
                    x_col, 
                    y_col, 
                    None if color_col == 'None' else color_col,
                    title=f"{y_col} by {x_col}",
                    orientation='v' if orientation == "Vertical" else 'h'
                )
                st.plotly_chart(bar_fig, use_container_width=True)
        
        elif chart_type == "Histogram":
            st.markdown("### Histogram")
            
            # Select column for histogram
            hist_col = st.selectbox("Select column", options=df.select_dtypes(include=['float64', 'int64']).columns)
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
            
            if hist_col:
                # Create histogram
                hist_fig = viz.create_histogram(
                    df, 
                    hist_col, 
                    bins=bins,
                    title=f"Distribution of {hist_col}"
                )
                st.plotly_chart(hist_fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot":
            st.markdown("### Scatter Plot")
            
            # Select columns for scatter plot
            x_col = st.selectbox("Select X-axis column", options=df.select_dtypes(include=['float64', 'int64']).columns)
            y_col = st.selectbox("Select Y-axis column", options=[col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != x_col])
            color_col = st.selectbox("Select color column (optional)", options=['None'] + list(df.columns))
            size_col = st.selectbox("Select size column (optional)", options=['None'] + list(df.select_dtypes(include=['float64', 'int64']).columns))
            
            if x_col and y_col:
                # Create scatter plot
                scatter_fig = viz.create_scatter_plot(
                    df, 
                    x_col, 
                    y_col, 
                    None if color_col == 'None' else color_col,
                    None if size_col == 'None' else size_col,
                    title=f"{y_col} vs {x_col}"
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
        
        elif chart_type == "Pie Chart":
            st.markdown("### Pie Chart")
            
            # Select columns for pie chart
            names_col = st.selectbox("Select category column", options=df.select_dtypes(include=['object']).columns)
            values_col = st.selectbox("Select values column", options=df.select_dtypes(include=['float64', 'int64']).columns)
            hole = st.slider("Donut hole size", min_value=0.0, max_value=0.8, value=0.4, step=0.1)
            
            if names_col and values_col:
                # Aggregate data for pie chart
                pie_data = df.groupby(names_col)[values_col].sum().reset_index()
                
                # Create pie chart
                pie_fig = viz.create_pie_chart(
                    pie_data, 
                    names_col, 
                    values_col, 
                    title=f"{values_col} by {names_col}",
                    hole=hole
                )
                st.plotly_chart(pie_fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            st.markdown("### Box Plot")
            
            # Select columns for box plot
            x_col = st.selectbox("Select category column", options=['None'] + list(df.select_dtypes(include=['object']).columns))
            y_col = st.selectbox("Select values column", options=df.select_dtypes(include=['float64', 'int64']).columns)
            color_col = st.selectbox("Select color column (optional)", options=['None'] + list(df.select_dtypes(include=['object']).columns))
            
            if y_col:
                # Create box plot
                box_fig = viz.create_box_plot(
                    df, 
                    None if x_col == 'None' else x_col, 
                    y_col, 
                    None if color_col == 'None' else color_col,
                    title=f"Distribution of {y_col}" + (f" by {x_col}" if x_col != 'None' else "")
                )
                st.plotly_chart(box_fig, use_container_width=True)
    
    # Tab 4: Time Series Analysis
    with tab4:
        st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)
        
        # Check if Date column exists and is datetime
        if 'Date' in df.columns and pd.api.types.is_datetime64_dtype(df['Date']):
            # Select column for time series analysis
            ts_col = st.selectbox(
                "Select column for time series analysis",
                options=df.select_dtypes(include=['float64', 'int64']).columns
            )
            
            # Select frequency for resampling
            freq = st.selectbox(
                "Select time frequency",
                options=[
                    ("Daily", "D"),
                    ("Weekly", "W"),
                    ("Monthly", "M"),
                    ("Quarterly", "Q"),
                    ("Yearly", "Y")
                ],
                format_func=lambda x: x[0]
            )
            
            if ts_col:
                # Perform time series analysis
                ts_data = sa.analyze_time_series(df, 'Date', ts_col, freq[1])
                
                if ts_data is not None:
                    # Create time series plot
                    ts_fig = viz.create_time_series_plot(
                        ts_data.reset_index(), 
                        'Date', 
                        f"{ts_col}_sum",
                        title=f"{ts_col} Over Time ({freq[0]})"
                    )
                    st.plotly_chart(ts_fig, use_container_width=True)
                    
                    # Show time series statistics
                    st.markdown("### Time Series Statistics")
                    st.dataframe(ts_data, use_container_width=True)
                    
                    # Calculate growth rates
                    if len(ts_data) > 1:
                        ts_data_growth = ts_data.copy()
                        ts_data_growth[f"{ts_col}_growth"] = ts_data_growth[f"{ts_col}_sum"].pct_change() * 100
                        
                        # Create growth rate plot
                        growth_fig = viz.create_time_series_plot(
                            ts_data_growth.reset_index(), 
                            'Date', 
                            f"{ts_col}_growth",
                            title=f"{ts_col} Growth Rate ({freq[0]})"
                        )
                        st.plotly_chart(growth_fig, use_container_width=True)
                else:
                    st.warning("Unable to perform time series analysis with the selected parameters")
        else:
            st.warning("Date column not found or not in datetime format. Time series analysis requires a valid date column.")
    
    # Tab 5: Insights
    with tab5:
        st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)
        
        # Generate insights based on the data
        with st.container():
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            
            # Basic data insights
            st.markdown(f"• Dataset contains <span class='highlight'>{len(df):,}</span> records with <span class='highlight'>{len(df.columns)}</span> columns.", unsafe_allow_html=True)
            
            # Date range insight
            if 'Date' in df.columns and pd.api.types.is_datetime64_dtype(df['Date']):
                date_range = f"{df['Date'].min().strftime('%d %b %Y')} to {df['Date'].max().strftime('%d %b %Y')}"
                st.markdown(f"• Data spans from <span class='highlight'>{date_range}</span>.", unsafe_allow_html=True)
            
            # Top categories insight
            for cat_col in df.select_dtypes(include=['object']).columns:
                if df[cat_col].nunique() < 20:  # Only for columns with reasonable number of categories
                    top_cats = df[cat_col].value_counts().head(3)
                    cats_str = ", ".join([f"{cat} ({count})" for cat, count in top_cats.items()])
                    st.markdown(f"• Top {cat_col}: <span class='highlight'>{cats_str}</span>", unsafe_allow_html=True)
            
            # Numeric column insights
            for num_col in df.select_dtypes(include=['float64', 'int64']).columns[:3]:  # Limit to first 3 numeric columns
                avg_val = df[num_col].mean()
                max_val = df[num_col].max()
                
                # Format based on magnitude
                if avg_val < 0.01 or avg_val > 1000:
                    avg_str = f"{avg_val:.2e}"
                else:
                    avg_str = f"{avg_val:.2f}"
                
                if max_val < 0.01 or max_val > 1000:
                    max_str = f"{max_val:.2e}"
                else:
                    max_str = f"{max_val:.2f}"
                
                st.markdown(f"• Average {num_col}: <span class='highlight'>{avg_str}</span>, Maximum: <span class='highlight'>{max_str}</span>", unsafe_allow_html=True)
            
            # Correlation insights
            if len(df.select_dtypes(include=['float64', 'int64']).columns) >= 2:
                corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
                
                # Find highest correlation (excluding self-correlations)
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr_idx = corr_matrix.abs().stack().idxmax()
                max_corr_val = corr_matrix.loc[max_corr_idx]
                
                corr_direction = "positive" if max_corr_val > 0 else "negative"
                st.markdown(f"• Strongest correlation: <span class='highlight'>{max_corr_idx[0]}</span> and <span class='highlight'>{max_corr_idx[1]}</span> have a {corr_direction} correlation of <span class='highlight'>{max_corr_val:.2f}</span>", unsafe_allow_html=True)
            
            # Time-based insights
            if 'Date' in df.columns and pd.api.types.is_datetime64_dtype(df['Date']) and len(df.select_dtypes(include=['float64', 'int64']).columns) > 0:
                # Find column with highest growth
                growth_insights = []
                
                for col in df.select_dtypes(include=['float64', 'int64']).columns[:3]:  # Limit to first 3 numeric columns
                    monthly_data = df.groupby(pd.Grouper(key='Date', freq='M'))[col].sum()
                    
                    if len(monthly_data) >= 2:
                        first_month = monthly_data.iloc[0]
                        last_month = monthly_data.iloc[-1]
                        
                        if first_month > 0:
                            growth_pct = (last_month - first_month) / first_month * 100
                            growth_insights.append((col, growth_pct))
                
                if growth_insights:
                    # Sort by absolute growth percentage
                    growth_insights.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_growth = growth_insights[0]
                    
                    growth_direction = "increased" if top_growth[1] > 0 else "decreased"
                    st.markdown(f"• <span class='highlight'>{top_growth[0]}</span> has {growth_direction} by <span class='highlight'>{abs(top_growth[1]):.1f}%</span> over the time period.", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Custom insights based on the dataset
            if 'Product Category' in df.columns and 'Total Revenue' in df.columns:
                st.markdown("### Product Category Analysis")
                
                # Top performing categories
                top_categories = df.groupby('Product Category')['Total Revenue'].sum().sort_values(ascending=False)
                
                # Create bar chart for top categories
                cat_fig = viz.create_bar_chart(
                    top_categories.reset_index(), 
                    'Product Category', 
                    'Total Revenue', 
                    title="Revenue by Product Category"
                )
                st.plotly_chart(cat_fig, use_container_width=True)
            
            if 'Region' in df.columns and 'Total Revenue' in df.columns:
                st.markdown("### Regional Analysis")
                
                # Revenue by region
                region_revenue = df.groupby('Region')['Total Revenue'].sum().sort_values(ascending=False)
                
                # Create pie chart for region revenue
                region_fig = viz.create_pie_chart(
                    region_revenue.reset_index(), 
                    'Region', 
                    'Total Revenue', 
                    title="Revenue by Region"
                )
                st.plotly_chart(region_fig, use_container_width=True)
            
            if 'Payment Method' in df.columns and 'Transaction ID' in df.columns:
                st.markdown("### Payment Method Analysis")
                
                # Transactions by payment method
                payment_counts = df.groupby('Payment Method')['Transaction ID'].count().sort_values(ascending=False)
                
                # Create pie chart for payment methods
                payment_fig = viz.create_pie_chart(
                    payment_counts.reset_index(), 
                    'Payment Method', 
                    'Transaction ID', 
                    title="Transactions by Payment Method"
                )
                st.plotly_chart(payment_fig, use_container_width=True)
else:
    # Show welcome message when no data is loaded
    st.markdown("""
    # Welcome to the Interactive Data Analysis Application
    
    This application allows you to:
    
    - Load and preprocess CSV data
    - Perform statistical analysis
    - Create interactive visualizations
    - Analyze time series data
    - Generate insights from your data
    
    To get started, please upload a CSV file using the sidebar or use the example data.
    """)

# Footer
st.markdown("---")
st.markdown(" 2025 Interactive Data Analysis Application | Created with Streamlit")