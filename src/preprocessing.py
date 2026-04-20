import os
import pandas as pd
import src.config as config

def preprocess_data(df):
    df = df.copy()

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    df = df.dropna()
    
    df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv(config.PROCESSED_PATH, index=False)
    
    return df