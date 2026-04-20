# Loan Default Risk - Statistical Study

## Overview
This project analyzes loan default risk using the German Credit Dataset. The objective is to statistically model the probability of default using logistic regression and interpret feature significance.

## Tech Stack
- NumPy
- pandas
- statsmodels
- scikit-learn
- matplotlib

## Problem Statement
Predict whether a customer is likely to default on a loan using structured financial and demographic data.

## Dataset
German Credit Dataset (UCI)
- Target: Credit Risk (Good / Bad)
- 'Bad' is treated as loan default

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