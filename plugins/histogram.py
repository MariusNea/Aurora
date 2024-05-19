#####################################################
#### Package: Aurora
#### Plugin: Histogram
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################


import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(df):
    """
    Plots histograms for each numerical column in the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to plot histograms for.
    """
    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")

    # Get the numerical columns from the DataFrame
    numerical_columns = df.select_dtypes(include='number').columns

    # Plot histograms for each numerical column
    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=30, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Example usage
def register(app):
    @app.register_plugin('statistics', 'histogram', 'Histogram')
    def histogram():
        datah = app.get_dataframe()

        plot_histogram(datah)
