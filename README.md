# Cloud Spark Data Processing Service

A cloud based data processing service using Apache Spark for machine learning analytics.

## What This Does

Upload your dataset and run machine learning algorithms on it using PySpark. The service computes statistics and runs 4 ML models:
- Linear Regression
- KMeans Clustering  
- Random Forest
- Correlation Analysis

## Live Demo

Streamlit App: https://cloud-spark-project.streamlit.app

Note: The app goes to sleep after being inactive for a while. If it shows a loading screen just wait a few seconds for it to wake up.

## How to Run

### Option 1: Use the Streamlit App

1. Go to https://cloud-spark-project.streamlit.app
2. Upload your results.json file from Google Drive
3. View your processed results

### Option 2: Run the Colab Notebook

1. Open the Colab notebook (link in the repo)
2. Run Cell 1 to install PySpark
3. Run Cell 3 and upload your data file (CSV, JSON, TXT)
4. Enter your options when asked:
   - Number of clusters (2-10)
   - Number of trees (10-100)
   - Test split percentage (10-40)
5. Run the remaining cells
6. Results save to your Google Drive automatically

## Files

- colab_notebook.py - Main processing code for Google Colab
- app.py - Streamlit web app for viewing results
- requirements.txt - Python dependencies
- covtype.csv - Sample dataset (Forest Cover Type)

## Requirements

For Streamlit: streamlit, pandas, numpy

For Colab: pyspark

## Dataset

Tested with Forest Cover Type dataset from UCI (581,012 rows, 55 columns).

You can use any CSV with numeric columns.

## Benchmark Results

Cores 1: 88.87 sec, 1.00x speedup
Cores 2: 79.41 sec, 1.12x speedup
Cores 4: 82.05 sec, 1.08x speedup
Cores 8: 91.33 sec, 0.97x speedup

Best performance at 2 cores because it matches Colab's physical CPU limit.
