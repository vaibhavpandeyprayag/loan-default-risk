import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess_data
# from src.model import train_lr
from src.model import train_dt
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.config import TARGET_COLUMN

def main():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = train_lr(X, y)
    model = train_dt(X_train, y_train)
    # Decision tree does not have a summary method like statsmodels
    # print(model.summary())

    metrics, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    print(metrics)

    # Feature Importance
    print("\nFeature Importance:")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    print(feature_importance.sort_values(ascending=False))

    # Visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)

    os.makedirs("reports", exist_ok=True)
    # with open("reports/model_summary.txt", "w") as f:
    #     f.write(model.summary().as_text())
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)
    
    results_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob
    })
    results_df.to_csv("reports/predictions.csv", index=False)

if __name__ == "__main__":
    main()