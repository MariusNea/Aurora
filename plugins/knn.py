#####################################################
#### Package: Aurora
#### Plugin: K Nearest Neighbors
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################


import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class KNNApp:
    def __init__(self, master, df):
        self.master = master
        self.df = df
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.features = None
        self.target = None
        
        # Text area for displaying information
        self.text_area = tk.Text(master, height=10, width=50)
        self.text_area.pack()

        # Button to train the KNN
        self.train_btn = tk.Button(master, text="Train KNN", command=self.open_feature_selection)
        self.train_btn.pack()

        # Button to make a prediction
        self.predict_btn = tk.Button(master, text="Make Prediction", command=self.make_prediction)
        self.predict_btn.pack()

    def open_feature_selection(self):
        # Opens a new window to select features
        self.feature_window = tk.Toplevel(self.master)
        self.feature_window.title("Select Features")
        
        tk.Label(self.feature_window, text="Enter features separated by commas:").pack()
        
        self.feature_entry = tk.Entry(self.feature_window, width=50)
        self.feature_entry.pack(pady=10)

        submit_btn = tk.Button(self.feature_window, text="Submit", command=self.train_knn)
        submit_btn.pack()

    def train_knn(self):
        features = self.feature_entry.get().replace(' ', '').split(',')
        if all(feature in self.df.columns for feature in features):
            self.features = self.df[features]
            self.target = self.df.iloc[:, -1]  # Assuming the last column is the target

            self.model.fit(self.features, self.target)
            self.text_area.insert(tk.END, "Model trained with features: {}\n".format(", ".join(features)))
            self.feature_window.destroy()
        else:
            messagebox.showerror("Error", "One or more features are invalid")

    def make_prediction(self):
        # Opens a new window for predictions
        if self.features is None:
            messagebox.showerror("Error", "Model is not trained yet")
            return

        self.pred_window = tk.Toplevel(self.master)
        self.pred_window.title("Make Prediction")
        self.entries = []

        for feature in self.features.columns:
            row = tk.Frame(self.pred_window)
            lbl = tk.Label(row, width=15, text=feature, anchor='w')
            ent = tk.Entry(row)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            lbl.pack(side=tk.LEFT)
            ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries.append(ent)
        
        submit_btn = tk.Button(self.pred_window, text="Submit", command=self.submit_prediction)
        submit_btn.pack()

    def submit_prediction(self):
        try:
            input_data = [float(entry.get()) for entry in self.entries]
            prediction = self.model.predict([input_data])[0]
            self.text_area.insert(tk.END, f"Prediction data: {input_data}\n")
            self.text_area.insert(tk.END, f"Belonging class: {prediction}\n")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        finally:
            self.pred_window.destroy()


def register(app):
    @app.register_plugin('machine_learning', 'knn', 'K Nearest Neighbors')
    def knn():
        datas = app.get_dataframe()
        root = tk.Tk()
        appl = KNNApp(root, datas)
        root.mainloop()