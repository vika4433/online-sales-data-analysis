# Interactive Data Analysis App using Streamlit

📝 App Name: **online-sales-data-analysis**
📁 Main File: `online-sales-data-analysis.py`
📄 Data File: `Online Sales Data.csv`

## 💡 Objective:

Build an interactive app using Streamlit to analyze and visualize the "Online Sales Data.csv" file. The app will include data loading, cleaning, statistical analysis, interactive visualizations, and download options for processed data.

---

The development process is broken into 10 clear steps, each describing a focused prompt for AI-based app generation in Windsurf IDE.

## 🔹 Step 1: Environment Setup

**Instruction Prompt:**

````python
Set up the development environment as follows:

- Use **Python 3.10.0**.
- Create a `requirements.txt` file with the following dependencies:

```text
pandas
numpy
streamlit
matplotlib
seaborn
````

- Create a main Python script named **`online-sales-data-analysis.py`** to serve as the entry point for the Streamlit app.
- Ensure the working directory includes the data file: `"Online Sales Data.csv"`.

---

## 🔹 Step 2: Load Data

**Instruction Prompt:**
Read the CSV file named "Online Sales Data.csv" using Pandas and display the entire dataset using Streamlit.

- Use `pd.read_csv()` to load the data.
- Display the data using `st.dataframe(df)` to show the full dataset in a scrollable table.
- Make sure the table supports both horizontal and vertical scrolling if the data exceeds the visible area.
- Add basic error handling to show a message if the file fails to load.

---

## 🔹 Step 3: Data Processing & Cleaning

**Instruction Prompt:**
Perform the following data processing and cleaning steps:

- Detect and remove outliers from numerical columns.
- Handle missing values:
  - Use mean or median imputation for numeric columns.
  - Use mode or remove rows for categorical columns as appropriate.
- Convert data types to appropriate formats (e.g., dates, numeric).
- Prepare the cleaned dataset for analysis.

---

## 🔹 Step 4: Statistical Analysis

**Instruction Prompt:**
Calculate and display the following statistical measures for relevant numerical columns in the dataset:

- Mean
- Median
- Standard Deviation
- Variance

Use Pandas or NumPy as needed to perform these computations.

Display the results in a clear and readable format using Streamlit.

---

## 🔹 Step 5: Data Visualization

**Instruction Prompt:**
Create visualizations to help users explore the dataset interactively.

Use Streamlit and either Matplotlib or Seaborn to:

- Show a line chart of total sales over time (e.g., by invoice date).
- Show a bar chart of the top-selling products.
- Display a histogram or distribution plot for a selected numeric column.

Allow the user to select which column to visualize using a dropdown (`st.selectbox`).
Make the charts interactive and responsive.

---

## 🔹 Step 6: Display Processed Data

**Instruction Prompt:**
Display the fully processed and cleaned dataset in a scrollable interactive table using Streamlit.

- Use `st.dataframe()` to present the final DataFrame.
- Ensure both vertical and horizontal scrolling are supported if the data is large.
- You may optionally add column filters or sorting functionality.

---

## 🔹 Step 7: CSV Download Option

**Instruction Prompt:**
Add a download button in the Streamlit app to allow users to download the cleaned and processed dataset as a CSV file.

- Use `st.download_button()` to enable the CSV export.
- Convert the DataFrame to CSV using `df.to_csv(index=False)`.
- Name the file "processed_sales_data.csv".

---

## 🔹 Step 8: Code Documentation

**Instruction Prompt:**
Ensure that all code is well documented:

- Add explanatory comments above each function and logic block.
- Use inline comments to explain critical steps.
- Write clear and concise docstrings for functions (use triple quotes).
- Follow Python best practices for naming and structure.

## 🔹 Step 9: Final Touches

**Instruction Prompt:**

- Add a title and sidebar to the Streamlit app.
- Add explanations for each chart.
- Make sure the app layout is clean and easy to use.

---

## 🔹 Step 10: Running the Application

**Instruction Prompt:**

- Provide instructions on how to run the Streamlit app after development.
- The main application file should be named `online-sales-data-analysis.py`.
- Generate a `launch.json` configuration file to run the app in the development environment.

## ✅ Final Outcome:

An interactive Streamlit app that loads, cleans, analyzes, and visualizes the "Online Sales Data.csv" file with options to download the cleaned version.
