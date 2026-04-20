import os
import json
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
# from src.eda import plot_target_distribution
from src.model import train_lr
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.config import TARGET_COLUMN

def main():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    model = train_lr(X, y)
    print(model.summary())

    metrics, y_pred, y_pred_prob = evaluate_model(model, X, y)
    print(metrics)

    # 📊 Visualizations
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_pred_prob)

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_summary.txt", "w") as f:
        f.write(model.summary().as_text())
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)
    
    results_df = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob
    })
    results_df.to_csv("reports/predictions.csv", index=False)

if __name__ == "__main__":
    main()