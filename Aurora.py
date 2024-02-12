#!/usr/bin/env python
# coding: utf-8

#####################################################
#### Package: Aurora
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import tkinter as tk
from tkinter import ttk, StringVar
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
from tkinter import simpledialog
import os
import importlib.util
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Label, Entry, Button
from PIL import Image, ImageTk
from io import BytesIO
from matplotlib.widgets import RectangleSelector



class DataFrameEditor:
    def __init__(self, root, dataframe):
        self.root = root
        self.root.title("Aurora")
        self.dataframe = dataframe
        self.plugins = {}
        self.selected_columns = []
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        self.create_menu()
        
        self.tree = ttk.Treeview(root)
        self.tree.pack(expand=True, fill='both')

        self.setup_tree_view()
        self.add_controls()
        self.target_col = None
        self.model = None
        self.input_data = None
        highlighted1 = []
        highlighted2 = []
        self.sel_list = []
        
    def register_plugin(self, category, name, menu_text):
        def decorator(func):
            if name not in self.plugins:
                self.plugins[name] = func
                if category == 'statistics':
                    self.add_plugin_menu_item(self.stats_menu, menu_text, func)
                elif category == 'machine_learning':
                    self.add_plugin_menu_item(self.ml_menu, menu_text, func)
            else:
                print(f"Plugin '{name}' is already registered.")
            return func
        return decorator
    
    def create_menu(self):
        # Create a menu bar
        

        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        # Add menu items to the File menu
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)

        # Edit menu
        #edit_menu = tk.Menu(menu_bar, tearoff=0)
        #menu_bar.add_cascade(label="Edit", menu=edit_menu)
        #edit_menu.add_command(label="Cut", command=self.dummy_function)
        #edit_menu.add_command(label="Copy", command=self.dummy_function)
        #edit_menu.add_command(label="Paste", command=self.dummy_function)

        # Statistics menu
        self.stats_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Statistics", menu=self.stats_menu)
        self.stats_menu.add_command(label="Generate Statistics", command=self.dummy_function)
        self.stats_menu.add_command(label="Statistical Models", command=self.regressions)
        self.stats_menu.add_command(label="Time Series Decomposition", command=self.decompose_and_plot)
        #Machine Learning menu
        self.ml_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Machine Learning", menu=self.ml_menu)
        
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="License", command=self.show_license)
        
    def add_plugin_menu_item(self, menu, text, command):
        menu.add_command(label=text, command=command)
        
    def setup_tree_view(self):
    # Clear existing columns and rows in the Treeview
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree["columns"] = list(self.dataframe.columns)
        self.tree["show"] = "headings"
        for column in self.dataframe.columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, anchor='center')

    # Inserting rows from the updated DataFrame
        for _, row in self.dataframe.iterrows():
            self.tree.insert('', 'end', values=list(row))


    def add_controls(self):
        add_row_button = tk.Button(self.root, text="Add Row", command=self.add_row)
        add_row_button.pack(side='left')

        delete_row_button = tk.Button(self.root, text="Delete Row", command=self.delete_row)
        delete_row_button.pack(side='left')

        add_column_button = tk.Button(self.root, text="Add Column", command=self.add_column)
        add_column_button.pack(side='left')

        delete_column_button = tk.Button(self.root, text="Delete Column", command=self.delete_column)
        delete_column_button.pack(side='left')
        
        clear_button = tk.Button(self.root, text="Clear Selection", command=self.clear_list)
        clear_button.pack(side='right')
        
        # Button to plot selected columns
        plot_button = tk.Button(self.root, text="Plot or Brush", command=self.int_hig_wrap)
        plot_button.pack(side='right')
        
        # Button to select columns for plotting
        select_button = tk.Button(self.root, text="Select Columns to Plot or Brush", command=self.select_columns)
        select_button.pack(side='right')    

        self.tree.bind('<Double-1>', self.on_item_double_click)
        
    def clear_list(self):
        self.sel_list.clear()
        print(self.sel_list)
    def dummy_function(self):
        summary = self.dataframe.describe()
        result = "Summary Statistics"

        # Create a new window to display the result
        result_window = tk.Toplevel(self.root)
        result_window.title("summary Statistics")

        # Create a label to display the result
        result_label = tk.Label(result_window, text=summary, padx=10, pady=10)
        result_label.pack()
        
        
    def decompose_and_plot(self):
    
    # Function to handle plotting with the entered period
        def plot_with_period():
            try:
                period = int(period_entry.get())
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter a valid integer for the period. First column hast to be Date and second Series")
                return

        # Assuming the first column is datetime and the second column is values
            time_series = self.dataframe.iloc[:, 1]

        # Perform STL decomposition
            decomposition = STL(time_series, period=period).fit()

        # Extract components
            original = time_series
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

        # Plot the components
            root = tk.Toplevel(self.root)
            
            root.title("Time Series Decomposition")

            fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

            axs[0].plot(original, label='Original')
            axs[0].set_ylabel('Original')

            axs[1].plot(trend, label='Trend', color='orange')
            axs[1].set_ylabel('Trend')

            axs[2].plot(seasonal, label='Seasonal', color='green')
            axs[2].set_ylabel('Seasonal')

            axs[3].plot(residual, label='Residual', color='red')
            axs[3].set_ylabel('Residual')

            for ax in axs:
                ax.legend()

        # Embed the matplotlib plot into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            tk.mainloop()

    # Create a new window for period input
        period_window = tk.Toplevel(self.root)
        period_window.title("Enter Seasonality Period")

    # Label and Entry for period input
        label = Label(period_window, text="Enter Seasonality Period:")
        label.pack(pady=10)
        period_entry = Entry(period_window)
        period_entry.pack(pady=10)

    # Button to trigger the plot with the entered period
        plot_button = Button(period_window, text="Plot", command=plot_with_period)
        plot_button.pack(pady=10)
        
    def train_linear_regression(self, target_col):
        if len(self.dataframe) == 1:
            return None  # Return None if there's only one sample
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe.drop(columns=[target_col]), self.dataframe[target_col], test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_logistic_regression(self, target_col):
        if len(self.dataframe) == 1:
            return None  # Return None if there's only one sample
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe.drop(columns=[target_col]), self.dataframe[target_col], test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def train_decision_tree(self, target_col):
        if len(self.dataframe) == 1:
            return None  # Return None if there's only one sample
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe.drop(columns=[target_col]), self.dataframe[target_col], test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model

    def make_predictions(self, model, input_data):
        if model is None:
            return None  # Return None if the model is not trained
        predictions = model.predict(input_data)
        return predictions
    
    def on_predict_button_click(self, selected_model, entry_features, label_predictions):
    # Get values from entry widgets
        feature_values = [float(entry.get()) for entry in entry_features]

    # Create a DataFrame for prediction
        new_data = pd.DataFrame([feature_values], columns=self.dataframe.columns[:-1])

    # Train the selected model
        if selected_model == "linear":
            model = self.train_linear_regression(target_col='target')
        elif selected_model == "logistic":
            model = self.train_logistic_regression(target_col='target')
        elif selected_model == "tree":
            model = self.train_decision_tree(target_col='target')
        else:
            model = None

    # Make predictions
        if model is not None:
            predictions = self.make_predictions(model, new_data)
        # Display predictions in labels or handle as needed
            label_predictions.config(text=f"Prediction: {predictions}")
        else:
            label_predictions.config(text="Please select a valid model before predicting.")
    
    def regressions(self):
    # Create a Tkinter window
        window = tk.Toplevel(self.root)
        window.title("Machine Learning Predictions")

    # Ask the user for the number of features
        num_features = simpledialog.askinteger("Number of Features", "Enter the number of features(number of columns from 1 to n-1). Last column is the predicted column:")

    # Create entry widgets for user input features
        entry_features = []
        for i in range(num_features):
            entry = tk.Entry(window, width=10)
            entry.grid(row=i, column=1, padx=10, pady=10)
            entry_features.append(entry)
            label = tk.Label(window, text=f"Feature {i + 1}:")
            label.grid(row=i, column=0, padx=10, pady=10, sticky=tk.E)

    # Create radio buttons for selecting the model
        # Create radio buttons for selecting the model
        selected_model = tk.StringVar()
        linear_radio = tk.Radiobutton(window, text="Linear Regression", variable=selected_model, value="linear")
        linear_radio.grid(row=num_features, column=0, columnspan=2, pady=10)
        logistic_radio = tk.Radiobutton(window, text="Logistic Regression", variable=selected_model, value="logistic")
        logistic_radio.grid(row=num_features + 1, column=0, columnspan=2, pady=10)
        decision_tree_radio = tk.Radiobutton(window, text="Decision Tree", variable=selected_model, value="tree")
        decision_tree_radio.grid(row=num_features + 2, column=0, columnspan=2, pady=10)

    # Create labels for displaying predictions
        label_predictions = tk.Label(window, text="Predictions:")
        label_predictions.grid(row=num_features + 4, column=0, columnspan=2)

    # Create a button to trigger predictions
        predict_button = tk.Button(window, text="Predict", command=lambda: self.on_predict_button_click(selected_model.get(), entry_features, label_predictions))
        predict_button.grid(row=num_features + 3, column=0, columnspan=2, pady=10)


    # Placeholder DataFrame with an unknown number of columns
        data = {'target': [0]}  
        for i in range(num_features):
            data[f'feature{i + 1}'] = [0.0]  # Initialize with placeholder values

    # Start the Tkinter event loop
        window.mainloop()
        
    
    def show_about(self):
        messagebox.showinfo("About", "Aurora \nVersion 0.1\n\nCreated by Marius Neagoe\n\n www.mariusneagoe.com")
    
    def show_license(self):
        license_window = tk.Toplevel()
        license_window.title("License")
        license_window.geometry("500x300")  # You can adjust the size as needed

    # Create a Text widget for displaying the license
        license_text_widget = tk.Text(license_window, wrap="word")
        license_text_widget.pack(expand=True, fill="both", padx=10, pady=10)

    # License text
        license_text = """ AURORA - Accessible User-friendly Resources for Optimized Research Analytics
Copyright (C) 2024 Marius Neagoe (www.mariusneagoe.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. 

"""
        
    # Insert the license text into the Text widget and disable editing
        license_text_widget.insert(tk.END, license_text)
        license_text_widget.config(state="disabled")
    
    
    def select_columns(self):
        # Use simpledialog to prompt the user for column selection
        selected_columns = simpledialog.askstring("Select Columns", "Enter two column names separated by a comma (e.g., col1, col2):")
        if selected_columns:
            columns = [col.strip() for col in selected_columns.split(',')]
            if len(columns) == 2:
                self.selected_columns = columns
                self.sel_list.append(selected_columns)
                
            else:
                messagebox.showerror("Error", "Please enter exactly two column names.")
                self.select_columns()
                
    def int_hig_wrap(self):
        date1, high = self.sel_list[0].split(', ')
        
        if len(self.sel_list) == 1:
            #messagebox.showinfo("Info", "Press OK to plot. You have to select 2 pairs of columns in order to Brush.")
            plt.figure(figsize=(10, 6))
            plt.scatter(self.dataframe[date1], self.dataframe[high])
            plt.xlabel(date1)
            plt.ylabel(high)
            plt.show()
        else:
            date2, target = self.sel_list[1].split(', ')
            col1 = date1
            col2 = high
            col3 = date2
            col4 = target
            self.interactive_highlight(col1, col2, col3, col4)
            
    def interactive_highlight(self, col1, col2, col3, col4):
    
    # Proceed with the interactive highlight functionality for non-empty col3 and col4
    # Check for identical columns among col1, col2, col3, col4
        cols = [col1, col2, col3, col4]
        distinct_values = []
        seen_values = set()

        for value in cols:
            if value not in seen_values:
                seen_values.add(value)
                distinct_values.append(value)
                
        x_col, y1_col, y2_col = distinct_values
    # Plotting both graphs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        line1, = ax1.plot(self.dataframe[x_col], self.dataframe[y1_col], 'ro', picker=5)
        line2, = ax2.plot(self.dataframe[x_col], self.dataframe[y2_col], 'bo')

        highlighted1 = []
        highlighted2 = []

        def clear_previous_highlights():
            for hl in highlighted1:
                hl.remove()
            highlighted1.clear()
            for hl in highlighted2:
                hl.remove()
            highlighted2.clear()

        def onselect(eclick, erelease):
            clear_previous_highlights()
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            mask = (self.dataframe[x_col] >= min(x1, x2)) & (self.dataframe[x_col] <= max(x1, x2)) & \
                   (self.dataframe[y1_col] >= min(y1, y2)) & (self.dataframe[y1_col] <= max(y1, y2))
            selected = self.dataframe[mask]
            hl1 = ax1.plot(selected[x_col], selected[y1_col], 'yo', linestyle='None', zorder=5)
            highlighted1.extend(hl1)
            hl2 = ax2.plot(selected[x_col], selected[y2_col], 'yo', linestyle='None', zorder=5)
            highlighted2.extend(hl2)
            fig.canvas.draw_idle()
            
        toggle_selector = RectangleSelector(ax1, onselect, useblit=True,
                                            button=[1],
                                            minspanx=5, minspany=5,
                                            spancoords='pixels',
                                            interactive=True)
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y1_col)
        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y2_col)
        
        plt.show()       
    

    def get_dataframe(self):
        return self.dataframe
    

    def on_item_double_click(self, event):
        item = self.tree.selection()[0]  # This gets the ID of the selected item in the Treeview
        column = self.tree.identify_column(event.x)  # Identifies the clicked column
        col_index = int(column.replace('#', '')) - 1  # Convert column ID to index
    
        new_value = simpledialog.askstring("Input", f"Enter new value:", parent=self.root)
        if new_value is not None:
            try:
                df_index = self.tree.index(item)  # Assuming direct correspondence between Treeview and DataFrame indices
                if df_index < len(self.dataframe):
                    self.dataframe.iat[df_index, col_index] = new_value  # Update DataFrame
                    self.tree.set(item, column=col_index, value=new_value)  # Update Treeview
                else:
                    print(f"Index {df_index} is out of bounds for the DataFrame.")
            except IndexError as e:
                print(f"Error updating cell: {e}")


    def add_row(self):
        new_row_index = len(self.dataframe)  # Next row index
        self.dataframe.loc[new_row_index] = [None] * len(self.dataframe.columns)  # Initialize new row with None or suitable defaults
        self.tree.insert('', 'end', values=([None] * len(self.dataframe.columns)))  # Add new row to Treeview as well


    def delete_row(self):
        selected_item = self.tree.selection()[0]  # Treeview's selected item ID
        if selected_item:
        # Assuming the order of items in the Treeview matches the DataFrame's index order
            index_in_df = self.tree.index(selected_item)  # Get index of the item in the Treeview
            df_index_to_delete = self.dataframe.index[index_in_df]  # Get corresponding DataFrame index
            self.dataframe.drop(df_index_to_delete, inplace=True)  # Drop the row from the DataFrame
            self.tree.delete(selected_item)  # Delete the item from the Treeview


    def add_column(self):
        new_column_name = simpledialog.askstring("Input", "Enter new column name:", parent=self.root)
        if new_column_name:
            self.dataframe[new_column_name] = ""
            self.setup_tree_view()

    def delete_column(self):
        column_name = simpledialog.askstring("Input", "Enter column name to delete:", parent=self.root)
        if column_name and column_name in self.df.columns:
        # Drop the column from the DataFrame
            self.dataframe.drop(columns=[column_name], inplace=True)
        
        # Rebuild the Treeview to reflect the change
            self.rebuild_treeview()

    def rebuild_treeview(self):
    # Clear the existing columns and data in the Treeview
        for col in self.tree['columns']:
            self.tree.delete(*self.tree.get_children())
            self.tree.heading(col, text='')
            self.tree.column(col, width=0, minwidth=0)

    # Setup the Treeview again with the updated DataFrame
        self.setup_tree_view()

        
def load_plugins(directory: str, app):
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('__'):
            plugin_path = os.path.join(directory, filename)
            module_name = os.path.splitext(filename)[0]
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Check if the module has a register function and call it with the app instance
            if hasattr(module, 'register'):
                module.register(app)        
        
        
    

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    root.iconphoto(False, tk.PhotoImage(file='icon.png'))
    # Use a file dialog to get the initial CSV file path
    initial_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not initial_file_path:
        messagebox.showinfo("Info", "No file selected. Exiting.")
        root.destroy()
        quit()
    try:
        # Load the initial CSV file into a DataFrame
        initial_df = pd.read_csv(initial_file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading initial CSV file: {e}")
        root.destroy()  # destroy the root window in case of an error
        quit()

    app = DataFrameEditor(root, initial_df)
    try:
        load_plugins('plugins', app)
    except RuntimeError as error:
        print(error)
        print("Some plugins did not load correctly and it may not work.")
        pass
    root.mainloop()