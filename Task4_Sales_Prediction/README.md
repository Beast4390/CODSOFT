# Sales Prediction Using Python

## Project Overview

Sales Prediction Using Python is a Machine Learning project that predicts product sales based on advertising expenditures across different media channels. The project uses Linear Regression to learn the relationship between advertising budgets and sales and helps businesses make data-driven marketing decisions.

## Problem Statement

Companies invest significant amounts in advertising through TV, Radio, and Newspaper channels. Predicting future sales based on advertising expenditure helps organizations optimize marketing budgets and improve profitability.

The objective of this project is to build a Machine Learning model capable of predicting sales using historical advertising data.

## Dataset

The dataset contains the following columns:

| Column    | Description                           |
| --------- | ------------------------------------- |
| TV        | Advertising budget spent on TV        |
| Radio     | Advertising budget spent on Radio     |
| Newspaper | Advertising budget spent on Newspaper |
| Sales     | Product sales generated               |

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

## Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous values. In this project, it is used to predict sales based on advertising expenditures.

## Workflow

1. Data Loading
2. Data Inspection
3. Missing Value Analysis
4. Exploratory Data Analysis (EDA)
5. Correlation Analysis
6. Feature Selection
7. Train-Test Split
8. Linear Regression Model Training
9. Model Evaluation
10. Sales Prediction

## Data Visualizations

The following visualizations were generated:

* Correlation Heatmap
* Pair Plot
* Sales vs TV Advertising
* Sales vs Radio Advertising
* Sales vs Newspaper Advertising
* Actual vs Predicted Sales Graph

## Model Evaluation Metrics

The model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Results

* MAE: 1.2748
* R² Score: 0.9059
* Predicted Sales: 15.54

The model achieved an R² Score of approximately 0.91, indicating strong predictive performance.

## Project Structure

Task4_Sales_Prediction

├── advertising.csv

├── sales_prediction.py

├── README.md

├── requirements.txt

├── output

│   └── model_output.txt

├── plots

│   ├── correlation_heatmap.png

│   ├── pairplot.png

│   ├── sales_vs_tv.png

│   ├── sales_vs_radio.png

│   ├── sales_vs_newspaper.png

│   └── actual_vs_predicted.png

└── sales_model.pkl

## Installation

Install the required libraries:

pip install -r requirements.txt

## Execution

Run the project:

python sales_prediction.py

## Future Enhancements

* Deploy using Streamlit
* Use advanced regression models
* Add real-time prediction interface
* Integrate cloud deployment

## Author

Nikhil Gembali

B.Tech – Computer Science and Design (CSD)

KIET
