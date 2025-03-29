# Online Sales Data Analysis

An interactive Streamlit application for analyzing and visualizing online sales data.

## Features

- Data loading and cleaning
- Statistical analysis of sales data
- Interactive visualizations
- Processed data download option

## Requirements

- Python 3.10.0
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run online-sales-data-analysis.py
```

This will start the Streamlit server and open the application in your default web browser.

## Application Structure

The application is divided into four main sections:

1. **Raw Data**: View the original dataset
2. **Processed Data**: View the cleaned dataset and download it as CSV
3. **Statistical Analysis**: View statistical measures for each column
4. **Visualizations**: Explore interactive charts and graphs

## Data Processing

The application performs the following data processing steps:

- Converts dates to datetime format
- Handles missing values using median/mode imputation
- Detects and caps outliers using the IQR method
- Provides statistical analysis for numeric and categorical columns

## Visualizations

The application includes several visualizations:

- Total sales over time
- Top 10 selling products
- Distribution of selected columns
- Sales by region
- Sales by payment method
- Sales by product category
