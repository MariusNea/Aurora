#plugins/poisson_probabilities.py
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import math

#####################################################
#### Package: Aurora
#### Plugin: Poisson Probabilities
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

## The dataframemust contain only one column which represents the number of events on a given period of time.
## Plugin outputh takes one argument, the numer of events for that period of time.
## Outputs 3 probabilities: the exact probability for exact that number of events to take place in the next period of time,
## the probability that < x number of events to take place in the next period of time,
## the probability that > x number of events to take place in the next period of time.

def poisson_probability(x, mu):
    return (math.exp(-mu) * (mu ** x)) / math.factorial(x)

def cumulative_poisson_probability(x, mu, cumulative=False):
    if cumulative:
        return sum(poisson_probability(i, mu) for i in range(x + 1))
    else:
        return poisson_probability(x, mu)

def real_world_poisson(mu, parameter, calculation_type='exact', **kwargs):
    if calculation_type == 'exact':
        return poisson_probability(parameter, mu)
    elif calculation_type == 'cumulative':
        return cumulative_poisson_probability(parameter, mu, cumulative=True)
    elif calculation_type == 'greater_than':
        return 1 - cumulative_poisson_probability(parameter, mu, cumulative=True)
    else:
        raise ValueError("Unsupported calculation type")

def calculate_and_display_results():
    # Calculate the average number of events per interval from the DataFrame
    mu = df[df.columns[0]].mean()
    
    x = simpledialog.askstring("Input", "Enter the specific number of events (x):")
    if x is not None:
        try:
            x = int(x)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer.")
            return
        
        # Calculate probabilities
        exact_probability = real_world_poisson(mu, x, 'exact')
        cumulative_probability = real_world_poisson(mu, x, 'cumulative')
        greater_than_probability = real_world_poisson(mu, x, 'greater_than')
        
        # Display results in a messagebox
        result_message = f"Exact Probability (x={x}): {exact_probability*100:.4f} %\n" \
                         f"Cumulative Probability (<=x): {cumulative_probability*100:.4f} %\n" \
                         f"Greater Than Probability (>x): {greater_than_probability*100:.4f} %"
        messagebox.showinfo("Probability Results", result_message)

def register(app):
    @app.register_plugin('statistics','poisson', 'Poisson Probabilities')
    def poisson():
        global df
        df = app.get_dataframe()
    # Check if the number of columns is even
        if len(df.columns) > 1:
            error_message = "Error: The number of columns in the dataframe must be 1. Each row represents the number of events on a period of time."
            messagebox.showerror("Error", error_message)
            return
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        calculate_and_display_results()

