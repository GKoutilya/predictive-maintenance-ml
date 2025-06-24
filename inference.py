import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

# Load saved scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('rul_model.pkl')

def preprocess(df):
    cols_to_scale = ['op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    scaled_data = scaler.transform(df[cols_to_scale])
    df_scaled_sensors = pd.DataFrame(scaled_data, columns=cols_to_scale)
    df_scaled = pd.concat([df[['unit', 'time']].reset_index(drop=True), df_scaled_sensors.reset_index(drop=True)], axis=1)

    for i in range(1, 22):
        sensor = f's{i}'
        df_scaled[f'{sensor}_roll_mean'] = df_scaled.groupby('unit')[sensor].transform(lambda x: x.rolling(window=5).mean())
        df_scaled[f'{sensor}_roll_std'] = df_scaled.groupby('unit')[sensor].transform(lambda x: x.rolling(window=5).std())

    flat_sensors = ['s1', 's5', 's10', 's16', 's18', 's19']
    df_scaled.drop(columns=flat_sensors, inplace=True)

    def add_slope_features(df, window=3):
        raw_sensor_cols = [col for col in df.columns if col.startswith('s') and '_roll' not in col]
        for sensor in raw_sensor_cols:
            df[f'{sensor}_delta'] = df.groupby('unit')[sensor].diff()
            df[f'{sensor}_slope'] = df.groupby('unit')[sensor].transform(lambda x: x.rolling(window=window).mean().diff())
        return df

    df_scaled = add_slope_features(df_scaled)
    df_scaled.dropna(inplace=True)

    return df_scaled

def predict_rul(df_new):
    df_processed = preprocess(df_new)
    X = df_processed.drop(columns=['unit', 'time'])
    predictions = model.predict(X)
    df_processed['predicted_RUL'] = predictions
    return df_processed[['unit', 'time', 'predicted_RUL']], df_processed, predictions

def save_evaluation_plots(y_true, y_pred, feature_importances, output_folder="evaluation_plots_inference"):
    os.makedirs(output_folder, exist_ok=True)

    if y_true is not None:
        # True vs Predicted RUL scatter plot
        scatter_path = os.path.join(output_folder, "true_vs_predicted_rul_inference.png")
        if not os.path.exists(scatter_path):
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel("True RUL")
            plt.ylabel("Predicted RUL")
            plt.title("True vs. Predicted Remaining Useful Life")
            plt.tight_layout()
            plt.savefig(scatter_path)
            plt.close()

        # Residuals histogram
        residuals_path = os.path.join(output_folder, "residuals_histogram_inference.png")
        if not os.path.exists(residuals_path):
            residuals = y_true - y_pred
            plt.figure(figsize=(8,5))
            sns.histplot(residuals, kde=True, bins=30, color='purple')
            plt.title("Distribution of Residual Errors")
            plt.xlabel("Error (True RUL - Predicted RUL)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(residuals_path)
            plt.close()
    else:
        print("True labels not provided, skipping evaluation plots involving true RUL.")

    # Feature importances bar plot (top 10) - always saved
    importances_path = os.path.join(output_folder, "feature_importances_inference.png")
    if not os.path.exists(importances_path):
        top_features = feature_importances.head(10)
        plt.figure(figsize=(10,5))
        sns.barplot(x=top_features.index, y=top_features.values, palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Top Feature Importances")
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(importances_path)
        plt.close()

if __name__ == "__main__":
    # Load new data (test dataset without true RUL)
    new_data_path = "CMaps/test_FD001.txt"
    column_names = ['unit', 'time', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    df_new = pd.read_csv(new_data_path, sep=r'\s+', header=None, names=column_names)

    results_df, processed_df, preds = predict_rul(df_new)
    print(results_df.head())
    results_df.to_csv("predicted_rul.csv", index=False)

    # No true RUL labels available here, so pass None for y_true
    feature_importances = pd.Series(model.feature_importances_, index=processed_df.drop(columns=['unit', 'time', 'predicted_RUL']).columns).sort_values(ascending=False)
    save_evaluation_plots(None, preds, feature_importances)