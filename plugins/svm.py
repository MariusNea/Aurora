#####################################################
#### Package: Aurora
#### Plugin: Support Vector Machines Classifier
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import threading


def preprocess_data(df, features, target):
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_svm(X_train, y_train, kernel='rbf', C=1.0):
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return accuracy, report

def create_gui(df):
    window = tk.Tk()
    window.title("SVM Classifier with Synthetic Data")

    # Determine initial values for features and target from DataFrame
    initial_features = ", ".join(df.columns[:-1])  # All columns except the last one
    initial_target = df.columns[-1]  # Last column

    kernel_var = tk.StringVar(window)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_var.set(kernel_options[2])  # default to RBF

    ttk.Label(window, text="Select Kernel:").pack(pady=5)
    kernel_dropdown = ttk.OptionMenu(window, kernel_var, kernel_options[2], *kernel_options)
    kernel_dropdown.pack(pady=5)

    results_text = scrolledtext.ScrolledText(window, width=60, height=10)
    results_text.pack(pady=10)

    global model, scaler
    model = None
    scaler = None

    def run_svm(features, target, kernel, C):
        global model, scaler
        try:
            features_list = [f.strip() for f in features.split(',')]
            X_scaled, y, scaler = preprocess_data(df, features_list, target.strip())
            X_train, X_test, y_train, y_test = split_data(X_scaled, y)
            model = train_svm(X_train, y_train, kernel, C)
            accuracy, report = evaluate_model(model, X_test, y_test)
            results_text.delete('1.0', tk.END)
            results_text.insert(tk.INSERT, f"Classification Report:\n{report}\n")
            results_text.insert(tk.INSERT, f"Accuracy: {accuracy:.2f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_input():
        features = simpledialog.askstring("Input", "Enter feature column names separated by comma:",
                                          initialvalue=initial_features)
        target = simpledialog.askstring("Input", "Enter target column name:", initialvalue=initial_target)
        kernel = kernel_var.get()
        C = float(simpledialog.askstring("Input", "Enter C parameter (e.g., 1.0):"))
        threading.Thread(target=run_svm, args=(features, target, kernel, C)).start()

    def make_prediction():
        if model is not None and scaler is not None:
            try:
                feature_inputs = simpledialog.askstring("Predict", "Enter values for {} separated by commas:".format(", ".join(df.columns[:-1])))
                if not feature_inputs:
                    messagebox.showwarning("Warning", "Input was cancelled or empty. Please provide valid numbers.")
                    return
                feature_values = [float(v.strip()) for v in feature_inputs.split(',')]
                if len(feature_values) != len(df.columns[:-1]):
                    messagebox.showerror("Error", "The number of input values must match the number of features.")
                    return
                data = pd.DataFrame([feature_values], columns=df.columns[:-1])
                scaled_data = scaler.transform(data)
                prediction = model.predict(scaled_data)
                messagebox.showinfo("Prediction Result", f"The predicted class is: {prediction[0]}")
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        else:
            messagebox.showerror("Error", "Model is not trained yet. Please train the model first.")

    btn_run = tk.Button(window, text="Train SVM", command=get_input)
    btn_run.pack(pady=10)

    btn_predict = tk.Button(window, text="Make Prediction", command=make_prediction)
    btn_predict.pack(pady=10)

    window.mainloop()

def register(app):
    @app.register_plugin('machine_learning', 'svm', 'Support Vector Machines')
    def svm():
        data = app.get_dataframe()
        create_gui(data)
