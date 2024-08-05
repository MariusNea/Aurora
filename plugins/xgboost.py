import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import messagebox

def xgboost_trainer_gui(data: pd.DataFrame):

    model = None
    feature_cols = None
    target_col = None
    prediction_feature_names = None

    def train_xgboost_model(data: pd.DataFrame, feature_cols=None, target_col=None, n_estimators=100, learning_rate=0.1, max_depth=5, early_stopping_rounds=10):
        
        if feature_cols is None or target_col is None:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        else:
            X = data[feature_cols]
            y = data[target_col]
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define the model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

        # Track early stopping manually
        min_val_error = float("inf")
        rounds_without_improvement = 0

        for i in range(n_estimators):
            model.n_estimators = i + 1  # Incrementally increase n_estimators
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            val_predictions = model.predict(X_val)
            val_error = mean_squared_error(y_val, val_predictions)

            if val_error < min_val_error:
                min_val_error = val_error
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement >= early_stopping_rounds:
                print(f"Early stopping after {i+1} rounds.")
                break
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return model, feature_importance, X.columns.tolist()

    def train_model():
        nonlocal model, feature_cols, target_col, prediction_feature_names

        if not entry_estimators.get().strip() or not entry_lr.get().strip() or not entry_depth.get().strip() or not entry_early_stopping_rounds.get().strip():
            messagebox.showwarning("Invalid Input", "All fields must not be empty.")
            return

        feature_cols = entry_features.get().strip()
        target_col = entry_target.get().strip()

        if not feature_cols:
            feature_cols = None
        else:
            feature_cols = [col.strip() for col in feature_cols.split(",")]

        if not target_col:
            target_col = None

        try:
            n_estimators = int(entry_estimators.get().strip())
            learning_rate = float(entry_lr.get().strip())
            max_depth = int(entry_depth.get().strip())
            early_stopping_rounds = int(entry_early_stopping_rounds.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter valid hyperparameters.")
            return

        model, feature_importance, feature_names = train_xgboost_model(data, feature_cols, target_col, n_estimators, learning_rate, max_depth, early_stopping_rounds)
        print("Model trained successfully!")
        print(feature_importance)

        # Store feature names for prediction
        prediction_feature_names = feature_names

        # Enable prediction inputs after training
        for entry in prediction_entries:
            entry.config(state='normal')
        btn_predict.config(state='normal')

    def predict():
        if model is None:
            messagebox.showwarning("Model Not Trained", "Please train the model before making predictions.")
            return

        try:
            input_data = [float(entry.get().strip()) for entry in prediction_entries]
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter valid numbers for predictions.")
            return

        input_df = pd.DataFrame([input_data], columns=prediction_feature_names)
        prediction = model.predict(input_df)[0]
        lbl_prediction_result.config(text=f"Predicted Value: {prediction:.2f}")

    # Tkinter GUI setup
    root = tk.Tk()
    root.title("XGBoost Model Trainer and Predictor")

    tk.Label(root, text="Feature Columns (comma separated):").pack()
    entry_features = tk.Entry(root, width=80)
    entry_features.pack(pady=5)

    tk.Label(root, text="Target Column:").pack()
    entry_target = tk.Entry(root, width=80)
    entry_target.pack(pady=5)

    tk.Label(root, text="Number of Estimators:").pack()
    entry_estimators = tk.Entry(root, width=20)
    entry_estimators.insert(0, "100")  # Default value
    entry_estimators.pack(pady=5)

    tk.Label(root, text="Learning Rate:").pack()
    entry_lr = tk.Entry(root, width=20)
    entry_lr.insert(0, "0.1")  # Default value
    entry_lr.pack(pady=5)

    tk.Label(root, text="Max Depth:").pack()
    entry_depth = tk.Entry(root, width=20)
    entry_depth.insert(0, "5")  # Default value
    entry_depth.pack(pady=5)

    tk.Label(root, text="Early Stopping Rounds:").pack()
    entry_early_stopping_rounds = tk.Entry(root, width=20)
    entry_early_stopping_rounds.insert(0, "10")  # Default value
    entry_early_stopping_rounds.pack(pady=5)

    btn_train_model = tk.Button(root, text="Train Model", command=train_model)
    btn_train_model.pack(pady=20)

    # Prediction Section
    tk.Label(root, text="Enter Values for Prediction:").pack(pady=10)
    prediction_entries = []
    for i in range(data.shape[1] - 1):  # Number of features
        entry = tk.Entry(root, width=20, state='disabled')
        entry.pack(pady=2)
        prediction_entries.append(entry)

    btn_predict = tk.Button(root, text="Predict", command=predict, state='disabled')
    btn_predict.pack(pady=20)

    lbl_prediction_result = tk.Label(root, text="")
    lbl_prediction_result.pack(pady=5)

    root.mainloop()


def register(app):
    @app.register_plugin('machine_learning', 'xgboost', 'XGBoost (Regression and Classification)')
    def xgboost():      
        date = app.get_dataframe()
        xgboost_trainer_gui(date)