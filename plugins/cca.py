

#####################################################
#### Package: Aurora
#### Plugin: Canonical Correlation Analysis
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def fill_na_with_mean(df):
    """Fill NaN values with the mean of their respective columns."""
    return df.fillna(df.mean())

def standardize_data(df):
    """Standardize DataFrame to have zero mean and unit variance."""
    df_filled = fill_na_with_mean(df)
    return (df_filled - df_filled.mean()) / df_filled.std()

def split_dataframe(df):
    """Split DataFrame into two equal halves."""
    mid_point = df.shape[1] // 2
    X = df.iloc[:, :mid_point]
    Y = df.iloc[:, mid_point:]
    return X, Y

def canonical_correlation_analysis(df):
    """Perform Canonical Correlation Analysis on a DataFrame."""
    X, Y = split_dataframe(df)
    X_std = standardize_data(X)
    Y_std = standardize_data(Y)
    
    S_xx = np.cov(X_std.T, bias=True)
    S_yy = np.cov(Y_std.T, bias=True)
    S_xy = np.cov(X_std.T, Y_std.T, bias=True)[:X_std.shape[1], X_std.shape[1]:]
    S_yx = S_xy.T

    # Ensure matrices are at least two-dimensional
    S_xx = np.atleast_2d(S_xx)
    S_yy = np.atleast_2d(S_yy)
    S_xy = np.atleast_2d(S_xy)
    S_yx = np.atleast_2d(S_yx)
    
    # Solve the generalized eigenvalue problem
    eigvals, eigvecs_x = eigh(S_xy @ np.linalg.inv(S_yy) @ S_yx, S_xx)
    eigvals = np.sqrt(np.maximum(eigvals, 0))  # Ensure non-negative eigenvalues
    
    idx = np.argsort(-eigvals)
    canonical_correlations = eigvals[idx]
    canonical_weights_x = eigvecs_x[:, idx]
    
    U = X_std @ canonical_weights_x
    V = Y_std @ (np.linalg.inv(S_yy) @ S_yx @ canonical_weights_x)
    
    return canonical_correlations, U, V
    
def plot_first_pair_canonical_variables(U, V):
    """
    Plot the first canonical variables from U and V against each other.
    U and V are the matrices of canonical variables, where each column is a canonical variable.
    This function focuses on the first pair, illustrating their relationship.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(U, V, edgecolor='k', alpha=0.7, label='Canonical Variable Pair')
    plt.title('Scatter Plot of the First Pair of Canonical Variables')
    plt.xlabel('First Canonical Variable from U')
    plt.ylabel('First Canonical Variable from V')
    plt.legend()
    plt.grid(True)
    plt.show()

def register(app):
    @app.register_plugin('statistics', 'cca', 'Canonical Correlation Analysis')
    def cca():
        data = app.get_dataframe()
		# You can add your code here
        canonical_correlations, U, V = canonical_correlation_analysis(data)
        print("Canonical Correlations:", canonical_correlations)
        print(U)
        print(V)
        plot_first_pair_canonical_variables(U, V)