# Credit Default Risk - Statistical Study

## Overview
This project analyzes credit default risk using the German Credit Dataset. The objective is to statistically model the probability of default using logistic regression and interpret feature significance.

## Tech Stack
- NumPy
- pandas
- statsmodels
- scikit-learn
- matplotlib

## Problem Statement
Predict whether a customer is likely to default on a credit using structured financial and demographic data.

## Dataset
German Credit Dataset
- Target: Credit Risk (Good / Bad)
- 1 - Good, 2 - Bad
- Link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## Project Pipeline
1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Logistic Regression Modeling (statsmodels)
5. Evaluation (scikit-learn)

## How to Run
```bash
conda env create -f environment.yml
conda activate credit-risk
python main.py
```

<!-- ## Performance Comparison Table


| Metric                | Accuracy            | Precision     | Recall        | ROC-AUC      |
|-----------------------|---------------------|---------------|---------------|--------------|
| Logistic Regression   | 68.0%               | 67.2%         | **54.1%**     | 0.72         |
| Decision Tree         | **70.1%**           | **78.2%**     | 45.0%         | **0.77**     |

## Business Interpretation Summary

- **Logistic Regression:**
    - Better at identifying default cases (higher recall)
    - Safer choice for financial institutions to minimize bad credits

- **Decision Tree:**
    - Higher Precision but lower Recall
    - May miss some default cases, but when it predicts default, it's more confident
 -->
