import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import datetime

# Set page configuration
st.set_page_config(
    page_title="Online Sales Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    """
    Load the sales data from CSV file
    Returns:
        DataFrame: The loaded sales data
    """
    try:
        df = pd.read_csv("data/Online Sales Data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to convert date strings to datetime objects
def convert_dates(df):
    """
    Convert date strings to datetime objects
    Args:
        df (DataFrame): The input dataframe
    Returns:
        DataFrame: DataFrame with converted date column
    """
    try:
        # Convert 'Date' column to datetime format (assuming DD/MM/YYYY format)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        return df
    except Exception as e:
        st.warning(f"Date conversion warning: {e}")
        return df

# Function to clean data
def clean_data(df):
    """
    Clean the data by handling missing values and outliers
    Args:
        df (DataFrame): The input dataframe
    Returns:
        DataFrame: Cleaned dataframe
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Convert date format
    cleaned_df = convert_dates(cleaned_df)
    
    # Identify numeric columns for outlier detection and imputation
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Handle missing values
    for col in numeric_cols:
        # Fill missing values with median for numeric columns
        if cleaned_df[col].isna().any():
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            st.info(f"Filled {cleaned_df[col].isna().sum()} missing values in '{col}' with median value: {median_val}")
    
    for col in non_numeric_cols:
        # Fill missing values with mode for categorical columns
        if cleaned_df[col].isna().any():
            mode_val = cleaned_df[col].mode()[0]
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            st.info(f"Filled {cleaned_df[col].isna().sum()} missing values in '{col}' with mode value: {mode_val}")
    
    # Detect and handle outliers in numeric columns using IQR method
    for col in numeric_cols:
        if col != 'Transaction ID':  # Skip Transaction ID as it's an identifier
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers instead of removing them
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                st.info(f"Capped {outliers} outliers in '{col}' column")
    
    return cleaned_df

# Function to calculate statistics
def calculate_statistics(df):
    """
    Calculate statistics for numeric columns
    Args:
        df (DataFrame): The input dataframe
    Returns:
        DataFrame: DataFrame with statistics
    """
    if df is None:
        return None
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Initialize statistics dictionary
    stats = {}
    
    # Calculate statistics for numeric columns
    for col in numeric_cols:
        if col != 'Transaction ID':  # Skip Transaction ID
            stats[col] = {
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Std Dev': df[col].std(),
                'Variance': df[col].var(),
                'Min': df[col].min(),
                'Max': df[col].max()
            }
    
    # For non-numeric columns, calculate count and unique values
    for col in non_numeric_cols:
        stats[col] = {
            'Count': df[col].count(),
            'Unique Values': df[col].nunique(),
            'Most Common': df[col].mode()[0] if not df[col].mode().empty else None
        }
    
    return stats

# Function to create visualizations
def create_visualizations(df, column=None):
    """
    Create visualizations based on the selected column
    Args:
        df (DataFrame): The input dataframe
        column (str): The column to visualize
    """
    if df is None:
        return
    
    # Convert dtype objects to strings for display
    column_types = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    
    # Line chart of total sales over time
    st.subheader("Total Sales Over Time")
    sales_by_date = df.groupby('Date')['Total Revenue'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sales_by_date['Date'], sales_by_date['Total Revenue'], marker='o', linestyle='-')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Revenue')
    ax.set_title('Total Sales Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.write("This chart shows the total sales revenue over time, helping identify sales trends and patterns.")
    
    # Bar chart of top-selling products
    st.subheader("Top 10 Selling Products")
    top_products = df.groupby('Product Name')['Units Sold'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_products.plot(kind='bar', ax=ax)
    ax.set_xlabel('Product Name')
    ax.set_ylabel('Units Sold')
    ax.set_title('Top 10 Selling Products')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("This chart displays the top 10 selling products based on the total number of units sold.")
    
    # Distribution plot for selected column
    if column:
        st.subheader(f"Distribution of {column}")
        
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(df[column], kde=True, ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {column}')
            plt.tight_layout()
            st.pyplot(fig)
            st.write(f"This histogram shows the distribution of values in the {column} column, helping identify patterns and outliers.")
        else:
            # For categorical columns, show a count plot
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts = df[column].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {column}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            st.write(f"This bar chart shows the distribution of categories in the {column} column.")

# Function to download data as CSV
def download_csv(df):
    """
    Create a download button for the dataframe
    Args:
        df (DataFrame): The dataframe to download
    """
    if df is None:
        return
    
    csv = df.to_csv(index=False)
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    return b

# Main application
def main():
    """
    Main function to run the Streamlit application
    """
    # Add title and description
    st.title("📊 Online Sales Data Analysis")
    st.markdown("""
    This application provides interactive analysis and visualization of online sales data.
    Use the sidebar to navigate between different sections of the app.
    """)
    
    # Create sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Raw Data", "Processed Data", "Statistical Analysis", "Visualizations"]
    )
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display raw data page
        if page == "Raw Data":
            st.header("Raw Data")
            st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.dataframe(df, use_container_width=True)
            
            # Show data types
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(dtype) for dtype in df.dtypes]
            })
            st.dataframe(dtypes_df, use_container_width=True)
        
        # Process and clean data
        cleaned_df = clean_data(df)
        
        # Display processed data page
        if page == "Processed Data":
            st.header("Processed Data")
            if cleaned_df is not None:
                st.write(f"Processed dataset contains {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns.")
                st.dataframe(cleaned_df, use_container_width=True)
                
                # Add download button
                st.download_button(
                    label="Download processed data as CSV",
                    data=download_csv(cleaned_df),
                    file_name="processed_sales_data.csv",
                    mime="text/csv"
                )
        
        # Display statistical analysis page
        if page == "Statistical Analysis":
            st.header("Statistical Analysis")
            if cleaned_df is not None:
                stats = calculate_statistics(cleaned_df)
                
                # Display statistics for each column
                for col, col_stats in stats.items():
                    with st.expander(f"Statistics for {col}"):
                        # Convert stats to DataFrame for better display
                        stats_df = pd.DataFrame.from_dict(col_stats, orient='index', columns=[col])
                        st.dataframe(stats_df)
        
        # Display visualizations page
        if page == "Visualizations":
            st.header("Data Visualizations")
            if cleaned_df is not None:
                # Create dropdown for column selection
                numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = cleaned_df.select_dtypes(exclude=['number']).columns.tolist()
                
                # Filter out Transaction ID from visualization options
                if 'Transaction ID' in numeric_cols:
                    numeric_cols.remove('Transaction ID')
                
                # Combine numeric and categorical columns
                all_cols = numeric_cols + categorical_cols
                
                # Create selectbox for column selection
                selected_column = st.selectbox(
                    "Select a column to visualize its distribution:",
                    all_cols
                )
                
                # Create visualizations
                create_visualizations(cleaned_df, selected_column)
                
                # Additional visualization: Sales by Region
                st.subheader("Sales by Region")
                region_sales = cleaned_df.groupby('Region')['Total Revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Region', y='Total Revenue', data=region_sales, ax=ax)
                ax.set_xlabel('Region')
                ax.set_ylabel('Total Revenue')
                ax.set_title('Total Sales by Region')
                plt.tight_layout()
                st.pyplot(fig)
                st.write("This chart shows the total sales revenue by region, helping identify the most profitable markets.")
                
                # Additional visualization: Sales by Payment Method
                st.subheader("Sales by Payment Method")
                payment_sales = cleaned_df.groupby('Payment Method')['Total Revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Payment Method', y='Total Revenue', data=payment_sales, ax=ax)
                ax.set_xlabel('Payment Method')
                ax.set_ylabel('Total Revenue')
                ax.set_title('Total Sales by Payment Method')
                plt.tight_layout()
                st.pyplot(fig)
                st.write("This chart shows the total sales revenue by payment method, helping identify preferred payment options.")
                
                # Additional visualization: Sales by Product Category
                st.subheader("Sales by Product Category")
                category_sales = cleaned_df.groupby('Product Category')['Total Revenue'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Product Category', y='Total Revenue', data=category_sales, ax=ax)
                ax.set_xlabel('Product Category')
                ax.set_ylabel('Total Revenue')
                ax.set_title('Total Sales by Product Category')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                st.write("This chart shows the total sales revenue by product category, helping identify the most profitable product categories.")

# Run the application
if __name__ == "__main__":
    main()
