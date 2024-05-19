#####################################################
#### Package: Aurora
#### Plugin: Text Classifier
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter.scrolledtext import ScrolledText  # Corrected import
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pandas as pd

class TextClassifierPlugin:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.parameters = {}
        self.status_text = None
        self.model = None

    def get_parameters_window(self):
        def submit():
            try:
                self.parameters['test_size'] = float(test_size_entry.get())
                self.parameters['random_state'] = int(random_state_entry.get())
                self.parameters['text_column'] = text_column_entry.get()
                self.parameters['label_column'] = label_column_entry.get()
                param_window.destroy()
            except ValueError:
                self.show_status("Invalid input. Please enter valid numbers for test size and random state, and column names.")

        param_window = tk.Toplevel(root)
        param_window.title("Set Parameters")

        tk.Label(param_window, text="Test Size (0-1):").grid(row=0, column=0)
        tk.Label(param_window, text="Random State:").grid(row=1, column=0)
        tk.Label(param_window, text="Text Column Name:").grid(row=2, column=0)
        tk.Label(param_window, text="Label Column Name:").grid(row=3, column=0)

        test_size_entry = tk.Entry(param_window)
        random_state_entry = tk.Entry(param_window)
        text_column_entry = tk.Entry(param_window)
        label_column_entry = tk.Entry(param_window)

        test_size_entry.grid(row=0, column=1)
        random_state_entry.grid(row=1, column=1)
        text_column_entry.grid(row=2, column=1)
        label_column_entry.grid(row=3, column=1)

        text_column_entry.insert(0, "Text")  # Default column name for text
        label_column_entry.insert(0, "Label")  # Default column name for labels

        submit_button = tk.Button(param_window, text="Submit", command=submit)
        submit_button.grid(row=4, columnspan=2)

        param_window.transient(root)
        param_window.grab_set()
        root.wait_window(param_window)

    def load_data(self):
        X = self.dataframe[self.parameters['text_column']]
        y = self.dataframe[self.parameters['label_column']]
        return X, y

    def train_model(self):
        self.show_status("Loading data...")
        X, y = self.load_data()

        self.show_status("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.parameters['test_size'], random_state=self.parameters['random_state'])

        self.show_status("Training model...")
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())
        self.model.fit(X_train, y_train)

        self.show_status("Evaluating model...")
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.show_status("Model trained. \n\n" + report)

    def make_prediction(self):
        input_text = self.prediction_entry.get()
        if self.model is not None:
            prediction = self.model.predict([input_text])
            self.show_status(f"Prediction for '{input_text}': {prediction[0]}")
        else:
            self.show_status("Model is not trained yet.")

    def show_status(self, message):
        if self.status_text:
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.config(state=tk.DISABLED)

    def main(self):
        global root
        root = tk.Tk()
        root.title("Text Classifier")

        self.status_text = ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(expand=True, fill='both')

        self.get_parameters_window()

        start_button = tk.Button(root, text="Start Training", command=self.train_model)
        start_button.pack()

        tk.Label(root, text="Enter text for prediction:").pack()
        self.prediction_entry = tk.Entry(root)
        self.prediction_entry.pack()

        predict_button = tk.Button(root, text="Make Prediction", command=self.make_prediction)
        predict_button.pack()

        root.mainloop()

def register(app):
    @app.register_plugin('machine_learning', 'text_classifier', 'Text Classifier')
    def kmeans():
        text = app.get_dataframe()

        classifier_plugin = TextClassifierPlugin(text)
        classifier_plugin.main()
