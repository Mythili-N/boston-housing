# Boston Housing Price Prediction

This project implements a machine learning workflow to predict housing prices using the Boston Housing dataset. It utilizes Linear Regression and Ridge Regression models to train and evaluate predictive performance.

## Features
- Loads the Boston Housing dataset from an online source.
- Prepares and processes the dataset into a structured DataFrame.
- Performs exploratory data analysis using a correlation heatmap.
- Splits data into training and testing sets.
- Trains Linear Regression and Ridge Regression models.
- Evaluates model performance using RMSE and R-squared metrics.
- Visualizes predictions using scatter plots.

## Requirements
Ensure you have Python installed along with the necessary dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
Run the script using:

```bash
python main.py
```

## File Structure
```
.
├── main.py         # Main script for model training and evaluation
├── README.md       # Project documentation
```

## Functions Overview
- `load_data(url)`: Loads and processes the dataset.
- `prepare_dataframe(data, target)`: Converts the dataset into a DataFrame.
- `split_data(df)`: Splits the dataset into training and testing sets.
- `create_correlation_heatmap(data)`: Generates a correlation heatmap.
- `train_models(X_train, y_train)`: Trains Linear and Ridge Regression models.
- `evaluate_model_performance(model, X_train, X_test, y_train, y_test)`: Evaluates a model’s performance.
- `evaluate_models(models, X_train, X_test, y_train, y_test)`: Evaluates multiple models.
- `plot_model_predictions(models, X_test, y_test)`: Plots model predictions against actual prices.
- `main()`: Orchestrates the workflow.

## Results
The script provides performance metrics for trained models, including:
- Root Mean Squared Error (RMSE)
- R-squared (R²) score

It also visualizes correlation heatmaps and model predictions.
