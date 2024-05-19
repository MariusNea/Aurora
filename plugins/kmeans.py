#####################################################
#### Package: Aurora
#### Plugin: K-Means
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import tkinter as tk
from tkinter import ttk, scrolledtext
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Function to run K-Means clustering
def run_kmeans(data, n_clusters, init_method, max_iter):
    print(f"Running KMeans with n_clusters={n_clusters}, init_method='{init_method}', max_iter={max_iter}")
    # Configure and run the KMeans algorithm
    kmeans = KMeans(
        n_clusters=int(n_clusters),
        init=init_method,
        max_iter=int(max_iter),
        algorithm='lloyd',
        random_state=42,
        n_init=10
    )
    kmeans.fit(data)
    return kmeans

# GUI creation function
def create_gui(data):
    # Root window
    root = tk.Tk()
    root.title("K-Means Clustering")

    # Entry for Number of Clusters
    tk.Label(root, text="Number of Clusters:").grid(row=0, column=0)
    n_clusters_entry = tk.Entry(root)
    n_clusters_entry.grid(row=0, column=1)

    # Dropdown for Initialization Methods
    tk.Label(root, text="Initialization Method:").grid(row=1, column=0)
    init_method_var = tk.StringVar(root)
    init_method_dropdown = ttk.Combobox(root, textvariable=init_method_var, state="readonly")
    init_method_dropdown['values'] = ('k-means++', 'random')
    init_method_dropdown.grid(row=1, column=1)
    init_method_dropdown.current(0)

    # Entry for Maximum Number of Iterations
    tk.Label(root, text="Max Iterations:").grid(row=2, column=0)
    max_iter_entry = tk.Entry(root)
    max_iter_entry.grid(row=2, column=1)

    # Scrolled Text Area for Output
    output_area = scrolledtext.ScrolledText(root, width=40, height=10)
    output_area.grid(row=5, column=0, columnspan=2, pady=10)

    # Button to Run K-Means
    def on_run_clicked():
        n_clusters = n_clusters_entry.get()
        init_method = init_method_var.get()
        max_iter = max_iter_entry.get()
        print(f"Button clicked with init_method='{init_method}'")

        if init_method not in ['k-means++', 'random']:
            output_area.delete('1.0', tk.END)
            output_area.insert(tk.INSERT, f"Invalid init method: {init_method}. Select 'k-means++' or 'random'.\n")
            return

        try:
            global model  # Declare model as global to use in prediction
            model = run_kmeans(data, int(n_clusters), init_method, int(max_iter))
            centers = model.cluster_centers_
            output = "Cluster Centers:\n{}\n".format(centers)
            output_area.delete('1.0', tk.END)
            output_area.insert(tk.INSERT, output)
        except Exception as e:
            output_area.delete('1.0', tk.END)
            output_area.insert(tk.INSERT, "Error: {}\n".format(e))

    run_button = tk.Button(root, text="Run K-Means", command=on_run_clicked)
    run_button.grid(row=4, column=0, columnspan=2)

    # Entry for Prediction Data
    tk.Label(root, text="Enter Prediction Data (comma-separated):").grid(row=6, column=0)
    prediction_entry = tk.Entry(root)
    prediction_entry.grid(row=6, column=1)

    # Button for Making Predictions
    def on_predict_clicked():
        prediction_data = prediction_entry.get()
        try:
            data_point = np.array([float(x) for x in prediction_data.split(',')]).reshape(1, -1)
            cluster = model.predict(data_point)
            output_area.insert(tk.END, "Predicted Cluster: {}\n".format(cluster[0]))
        except Exception as e:
            output_area.insert(tk.END, "Error in prediction: {}\n".format(e))

    predict_button = tk.Button(root, text="Make Prediction", command=on_predict_clicked)
    predict_button.grid(row=7, column=0, columnspan=2)

    # Start the GUI
    root.mainloop()

def register(app):
    @app.register_plugin('machine_learning', 'kmeans', 'Unsupervised Learning (K-Means)')
    def kmeans():
        dateq = app.get_dataframe()
        # Preprocess data: scaling
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(dateq)
        data_scaled = pd.DataFrame(data_scaled, columns=dateq.columns)
        # Running the GUI
        create_gui(data_scaled)
