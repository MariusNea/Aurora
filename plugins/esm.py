

#####################################################
#### Package: Aurora
#### Plugin: Exponential Smoothing Model
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def esf(df, column_name, period, trend, seasonal):
    """
    Applies Exponential Smoothing on a DataFrame's specified time series data column and plots the original data and forecast.
    
    :param df: DataFrame containing the time series data.
    :param column_name: Name of the column containing the time series data.
    :param period: The seasonal period.
    :param trend: The type of trend component ('additive', 'multiplicative', or None).
    :param seasonal: The type of seasonal component ('additive', 'multiplicative', or None).
    """
    # Validate column name
    if column_name not in df.columns:
        messagebox.showerror("Error", f"Column '{column_name}' not found in DataFrame.")
        return

    # Convert period to integer
    try:
        period = int(period)
    except Exception as e:
        # Display an error message box with the description of the exception
        messagebox.showerror("Error", f"An error occurred: {e}")
    
    # Convert 'None' strings to NoneType
    trend = None if trend == 'None' else trend
    seasonal = None if seasonal == 'None' else seasonal
    try:
    # Fit the model
        model = ExponentialSmoothing(df[column_name], trend=trend, seasonal=seasonal, seasonal_periods=period)
        model_fit = model.fit()
    except Exception as e:
        # Display an error message box with the description of the exception
        messagebox.showerror("Error", f"An error occurred: {e}")
        
    # Forecast
    forecast = model_fit.fittedvalues
    
    # Plot the original data and the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column_name], label='Original')
    plt.plot(df.index, forecast, label='Forecast', alpha=0.7)
    plt.title('Time Series Forecast')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


def register(app):
    @app.register_plugin('statistics', 'esm', 'Exponential Smoothing Model')
    def esm():
        data = app.get_dataframe()
		# Create the main window
        root = tk.Tk()
        root.title("Exponential Smoothing Parameters")

        # Column Name Entry
        tk.Label(root, text="Column Name:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        column_entry = tk.Entry(root)
        column_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')


        # Period Entry
        tk.Label(root, text="Period:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        period_entry = tk.Entry(root)
        period_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

        # Trend ComboBox
        tk.Label(root, text="Trend:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
        trend_options = ["additive", "multiplicative", "None"]
        trend_combobox = ttk.Combobox(root, values=trend_options, state="readonly")
        trend_combobox.grid(row=2, column=1, padx=10, pady=10, sticky='ew')
        trend_combobox.set("None")

        # Seasonal ComboBox
        tk.Label(root, text="Seasonal:").grid(row=3, column=0, padx=10, pady=10, sticky='w')
        seasonal_options = ["additive", "multiplicative", "None"]
        seasonal_combobox = ttk.Combobox(root, values=seasonal_options, state="readonly")
        seasonal_combobox.grid(row=3, column=1, padx=10, pady=10, sticky='ew')
        seasonal_combobox.set("None")

        # Submit Button
        submit_button = tk.Button(root, text="Submit", command=lambda: esf(data, column_entry.get(), period_entry.get(), trend_combobox.get(), seasonal_combobox.get()))

        submit_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Set the grid expansion properties
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(4, weight=1)

        root.mainloop()
        