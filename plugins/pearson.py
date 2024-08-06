#####################################################
#### Package: Aurora
#### Plugin: Pearson correlation
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

def calculate_pearson_correlation(df, col1, col2):
    """Calculate the Pearson correlation coefficient between two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns '{col1}' and '{col2}' are not in the DataFrame.")
    
    return df[col1].corr(df[col2])

def create_tkinter_ui(dataframe):
    """Create a Tkinter UI for inputting column names and calculating Pearson correlation."""
    def on_calculate():
        """Handler for the calculate button."""
        cols = entry.get().strip()
        try:
            if '-' in cols:
                parts = cols.split(',')
                if len(parts) != 2:
                    raise ValueError("Incorrect format. Use either 'col1,col2', 'col1,col2-col10', or 'col1-col100,col105-col200'.")

                range1 = parts[0].strip()
                range2 = parts[1].strip()

                if '-' in range1 and '-' in range2:
                    start_col1, end_col1 = range1.split('-')
                    start_col1 = start_col1.strip()
                    end_col1 = end_col1.strip()

                    start_col2, end_col2 = range2.split('-')
                    start_col2 = start_col2.strip()
                    end_col2 = end_col2.strip()

                    if start_col1 not in dataframe.columns or end_col1 not in dataframe.columns or start_col2 not in dataframe.columns or end_col2 not in dataframe.columns:
                        raise ValueError("One or more columns are not in the DataFrame.")
                    
                    # Get the range of columns for both parts
                    col_range1 = dataframe.loc[:, start_col1:end_col1].columns
                    col_range2 = dataframe.loc[:, start_col2:end_col2].columns

                    results = []
                    for col1 in col_range1:
                        for col2 in col_range2:
                            correlation = calculate_pearson_correlation(dataframe, col1, col2)
                            results.append((col1, col2, correlation))
                
                    # Convert results to a DataFrame for export
                    result_df = pd.DataFrame(results, columns=['Column 1', 'Column 2', 'Correlation'])
                    
                    # Save to CSV
                    save_results_to_csv(result_df)
                elif '-' in range1 or '-' in range2:
                    raise ValueError("Invalid range format. Both parts should be ranges if '-' is present in both.")
                else:
                    base_col = range1
                    other_col = range2
                    correlation = calculate_pearson_correlation(dataframe, base_col, other_col)
                    result_df = pd.DataFrame([(base_col, other_col, correlation)], columns=['Base Column', 'Compared Column', 'Correlation'])
                    
                    # Show the result in a message box
                    messagebox.showinfo("Pearson Correlation", f"The correlation between '{base_col}' and '{other_col}' is: {correlation:.4f}")
                    
                    # Save to CSV
                    save_results_to_csv(result_df)
            else:
                col1, col2 = cols.split(',')
                col1 = col1.strip()
                col2 = col2.strip()
                correlation = calculate_pearson_correlation(dataframe, col1, col2)
                result_df = pd.DataFrame([(col1, col2, correlation)], columns=['Base Column', 'Compared Column', 'Correlation'])
                
                # Show the result in a message box
                messagebox.showinfo("Pearson Correlation", f"The correlation between '{col1}' and '{col2}' is: {correlation:.4f}")
                
                # Save to CSV
                save_results_to_csv(result_df)
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_results_to_csv(result_df):
        """Save the correlation results to a CSV file."""
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            result_df.to_csv(file_path, index=False)
            messagebox.showinfo("Save Successful", f"Results saved to {file_path}")

    # Tkinter setup
    root = tk.Tk()
    root.title("Pearson Correlation Calculator")

    label = tk.Label(root, text="Enter column names (e.g., Col1,Col2 or Col1,Col2-Col10 or Col1-Col100,Col105-Col200):")
    label.pack()

    global entry  # Declare entry as global to access it within on_calculate
    entry = tk.Entry(root, width=50)
    entry.pack()

    button = tk.Button(root, text="Calculate Correlation", command=on_calculate)
    button.pack()

    root.mainloop()

def register(app):
    @app.register_plugin('statistics', 'pearson', 'Pearson Correlation')
    def pearson():      
        datafr = app.get_dataframe()
        create_tkinter_ui(datafr)
