import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(url):
    """Load the Boston Housing dataset from the given URL."""
    raw_df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target

def prepare_dataframe(data, target):
    """Prepare a DataFrame with features and target."""
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"
    ]
    df = pd.DataFrame(np.column_stack([data, target]), columns=columns)
    return df

def split_data(df):
    """Split the DataFrame into training and testing sets."""
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_correlation_heatmap(data):
    """Create and display a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def train_models(X_train, y_train):
    """Train Linear and Ridge Regression models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def evaluate_model_performance(model, X_train, X_test, y_train, y_test):
    """Evaluate a single model and return its performance metrics."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "Train RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "Train R^2": r2_score(y_train, train_pred),
        "Test R^2": r2_score(y_test, test_pred),
    }

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate multiple models and return their performance metrics."""
    return {name: evaluate_model_performance(model, X_train, X_test, y_train, y_test) for name, model in models.items()}

def plot_model_predictions(models, X_test, y_test):
    """Plot predictions of models against actual values."""
    plt.figure(figsize=(12, 8))
    for i, (name, model) in enumerate(models.items(), start=1):
        plt.subplot(2, 2, i)
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(name)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.tight_layout()
    plt.show()

def main():
    """Main function to orchestrate the workflow."""
    # Load data
    print("Loading and preparing data...")
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    data, target = load_data(data_url)
    df = prepare_dataframe(data, target)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Data shape:", df.shape)

    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    metrics = evaluate_models(models, X_train, X_test, y_train, y_test)
    print("\nModel Performance Metrics:")
    for model_name, metric in metrics.items():
        print(f"\n{model_name}:")
        for metric_name, value in metric.items():
            print(f"{metric_name}: {value:.4f}")

    # Create correlation heatmap
    print("\nCreating correlation heatmap...")
    create_correlation_heatmap(df)

    # Plot predictions
    print("\nPlotting predictions...")
    plot_model_predictions(models, X_test, y_test)

if __name__ == "__main__":
    main()
