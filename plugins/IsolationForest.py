# plugins/IsolationForest.py


#####################################################
#### Package: Aurora
#### Plugin: Outliers (Anomaly) Detection
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################
    
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.simpledialog import askstring
from tkinter import messagebox


def run_isolation_forest(df):
    col = ask_col()
    contamination = ask_contamination()  # Get contamination from the user    
    df_part = df[col]

    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=float(contamination), random_state=42)
    
    # Fit the model on the data
    # Note: .values.reshape(-1, 1) reshapes data for a single feature
    model.fit(df_part.values.reshape(-1, 1))
    
    # Predict outliers
    preds = model.predict(df_part.values.reshape(-1, 1))
    
    # Add predictions to the DataFrame
    df['outlier_' + col] = preds
    
    # Filter outliers
    outliers = df[df['outlier_' + col] == -1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    # Plot all data points, using a scatter plot
    # Y-values are zeros since it's a single dimension, with slight jitter added for visualization
    plt.scatter(df[col], [0 + jitter for jitter in preds * 0.02], c=preds, cmap='coolwarm', edgecolor='k', s=20)
    plt.title('Data Points Classified by Isolation Forest')
    plt.xlabel(col)  # X-axis label as the column name
    plt.yticks([])  # Hide Y-axis ticks since they are arbitrary
    plt.legend(['Inliers', 'Outliers'], loc='lower right')
    plt.savefig('outlier_plot' + '_column_' + col + '.png')
    plt.show()

def ask_contamination():
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    contamination = askstring("Input", "Enter the contamination factor (e.g., 0.01):", parent=root)
    root.destroy()

    return contamination    
    
def ask_col():
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    col = askstring("Input", "Enter column name on which tou want to perform outlier detection:", parent=root)
    root.destroy()

    return col
    
def register(app):
    @app.register_plugin('machine_learning', 'isolation_forest', 'Outliers (Anomaly) Detection')
    def isolation_forest():
        global df
        df = app.get_dataframe()
        run_isolation_forest(df)
        messagebox.showinfo("Results", "Your data was saved as a image in current folder.")
        


