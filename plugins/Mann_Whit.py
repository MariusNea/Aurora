# plugins/plugin_a.py
from scipy.stats import mannwhitneyu
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import pandas as pd

def register(app):
    @app.register_plugin('statistics','mann_whitney_u_test', 'Mann-Whitney U Test')
    def mann_whitney_u_test():
        df = app.get_dataframe()
    # Check if the number of columns is even
        if len(df.columns) % 2 != 0:
            error_message = "Error: The number of columns in the dataframe must be even. The test is done on the columns that are placed one next to another."
            messagebox.showerror("Error", error_message)
            return

        # Create tkinter window
        root = tk.Tk()
        root.title("Mann-Whitney U Test Results")

        # Create treeview to display results
        tree = ttk.Treeview(root)
        tree["columns"] = ("Column Pair", "U Statistic", "P-Value")

        # Define treeview columns
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("Column Pair", anchor=tk.W, width=100)
        tree.column("U Statistic", anchor=tk.W, width=100)
        tree.column("P-Value", anchor=tk.W, width=100)

        # Create treeview headings
        tree.heading("#0", text="", anchor=tk.W)
        tree.heading("Column Pair", text="Column Pair", anchor=tk.W)
        tree.heading("U Statistic", text="U Statistic", anchor=tk.W)
        tree.heading("P-Value", text="P-Value", anchor=tk.W)

        # Perform Mann-Whitney U test for adjacent column pairs
        for i in range(0, len(df.columns), 2):
            df['column1_clean'] = pd.to_numeric(df.iloc[:, i], errors='coerce')
            df['column2_clean'] = pd.to_numeric(df.iloc[:, i + 1], errors='coerce')
            df_clean = df.dropna(subset=['column1_clean', 'column2_clean'])
            result = mannwhitneyu(df_clean['column1_clean'], df_clean['column2_clean'])

        # Insert result into treeview
            tree.insert("", i, values=(f"{df.columns[i]} - {df.columns[i+1]}", result.statistic, result.pvalue))

        # Pack and run tkinter window
        tree.pack(expand=True, fill=tk.BOTH)
        root.mainloop()

