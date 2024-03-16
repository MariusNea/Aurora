
#####################################################
#### Package: Aurora
#### Plugin: One Way ANOVA
#### Version: 0.1
#### Author: Marius Neagoe
#### Copyright: © 2024 Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def validate_dataframe(dataframe):
    """
    Validates the DataFrame structure. Assumes the first column is categorical and the rest are numeric.
    Checks for at least one categorical column and at least two numeric columns.
    """
    # Check if the DataFrame has at least three columns (one categorical and at least two numeric)
    if dataframe.shape[1] < 3:
        raise ValueError("DataFrame must contain at least one categorical column and two numeric columns.")

    # Check if the first column is categorical (object or category dtype)
    if dataframe.dtypes[0] not in ['object', 'category']:
        raise ValueError("The first column must be categorical (type object or category).")
        exit()
    # Check if the remaining columns are numeric
    if not all(dataframe.dtypes[1:].apply(lambda dtype: np.issubdtype(dtype, np.number))):
        raise ValueError("All columns except the first must be numeric.")
        exit()
    # Check for missing values in the DataFrame
    if dataframe.isnull().any().any():
        print("Warning: DataFrame contains missing values. They will be handled appropriately.")

def handle_missing_values(dataframe):
    """
    Handles missing values by dropping rows with any missing values.
    """
    return dataframe.dropna()

def perform_anova_and_tukey(dataframe):
    """
    Performs ANOVA and Tukey's HSD test on the given DataFrame.
    Assumes the first column is categorical and the rest are numeric.
    """
    group_col = dataframe.columns[0]  # The first column as the categorical column
    numeric_cols = dataframe.columns[1:]  # The rest as numeric columns

    for col in numeric_cols:
        # Preparing groups for ANOVA
        groups = [dataframe[dataframe[group_col] == group][col].dropna() for group in dataframe[group_col].unique()]

        # Performing ANOVA
        f_stat, p_value = f_oneway(*groups)
        print(f"ANOVA result for {col}: F={f_stat}, p={p_value}")

        # If the p-value from ANOVA is significant, proceed with Tukey's HSD
        if p_value < 0.05:
            # Concatenating all group data into a single series for Tukey's test
            all_data = pd.concat(groups)
            all_groups = pd.concat([pd.Series([group] * len(g)) for group, g in zip(dataframe[group_col].unique(), groups)])

            # Performing Tukey's HSD test
            tukey = pairwise_tukeyhsd(endog=all_data, groups=all_groups, alpha=0.05)
            print(f"Tukey's HSD test result for {col}:\n{tukey}")
        else:
            print("ANOVA p-value > 0.05; Tukey's test not performed.")



def display_results(dataframe):
    """
    Orchestrates the analysis process, including validations, ANOVA, effect size calculation, 
    and graphical summary, then displays results and plots in a Tkinter window.
    """
    group_col = dataframe.columns[0]  # First column as categorical
    numeric_cols = dataframe.columns[1:]  # Remaining columns as numeric variables
    
    # Data Validation
    messagebox.showinfo("Preparation", "First column must be populated with categories that will take part on ANOVA. All other columns must be numeric.")
    validate_dataframe(dataframe)
    dataframe = handle_missing_values(dataframe)
    check_equal_variances(dataframe)
    check_normality(dataframe)
    calculate_effect_size(dataframe)
    # Perform ANOVA and Tukey's HSD Test
    perform_anova_and_tukey(dataframe)
    

def check_equal_variances(dataframe):
    """
    Checks for equal variances among groups using Levene's test for each numeric variable.
    
    :param dataframe: The pandas DataFrame containing the data.
    :param group_col: The name of the column containing the categorical variable.
    """
    group_col = dataframe.columns[0]  # First column as categorical
    numeric_cols = dataframe.columns[1:]  # Remaining columns as numeric variables
    
    for col in numeric_cols:
        print(f"Levene's test for {col}:")
        groups = [dataframe[dataframe[group_col] == group][col].dropna() for group in dataframe[group_col].unique()]
        statistic, p_value = stats.levene(*groups)
        if p_value < 0.05:
            print(f"    Warning: Unequal variances detected (p-value: {p_value:.3f}).")
        else:
            print(f"    Equal variances confirmed (p-value: {p_value:.3f}).")

def check_normality(dataframe):
    """
    Checks for normality in each group for each numeric variable using the Shapiro-Wilk test.
    
    :param dataframe: The pandas DataFrame containing the data.
    :param group_col: The name of the column containing the categorical variable.
    """
    group_col = dataframe.columns[0]  # First column as categorical
    numeric_cols = dataframe.columns[1:]  # Remaining columns as numeric variables
    
    for col in numeric_cols:
        print(f"Shapiro-Wilk test for normality in {col}:")
        for group in dataframe[group_col].unique():
            group_data = dataframe[dataframe[group_col] == group][col].dropna()
            statistic, p_value = stats.shapiro(group_data)
            if p_value < 0.05:
                print(f"    Group {group}: Non-normal distribution detected (p-value: {p_value:.3f}).")
            else:
                print(f"    Group {group}: Normal distribution confirmed (p-value: {p_value:.3f}).")

# Integrate these functions into your existing workflow as needed, calling them before conducting ANOVA.
def calculate_effect_size(dataframe):
    """
    Calculates the effect size (eta squared) for each numeric variable against the categorical variable.
    
    :param dataframe: The pandas DataFrame containing the data.
    :param group_col: The name of the column containing the categorical variable.
    :return: A dictionary with numeric columns as keys and their eta squared values as values.
    """
    eta_squared_values = {}
    group_col = dataframe.columns[0]  # First column as categorical
    numeric_cols = dataframe.columns[1:]  # Remaining columns as numeric variables
    
    for col in numeric_cols:
        formula = f"{col} ~ C({group_col})"
        model = ols(formula, data=dataframe).fit()
        aov_table = anova_lm(model, typ=2)
        
        ss_between = aov_table.sum_sq['C({})'.format(group_col)]  # Corrected access method
        ss_total = sum(aov_table.sum_sq)
        eta_squared = ss_between / ss_total
        eta_squared_values[col] = eta_squared
    print("Effect sizes of groups:")
    print(eta_squared_values)   

def register(app):
    @app.register_plugin('statistics', 'anova', 'One Way ANOVA')
    def anova():
        df = app.get_dataframe()

        display_results(df)
