AURORA - Problem solving focused statistical and machine learning software toolkit. 

In today's world, the fields of statistics and machine learning hold immense potential for solving real-world problems and significantly impacting businesses and daily life. However, the complexity and learning curve associated with these fields can be daunting, making it challenging for those interested to effectively utilize these tools. Recognizing this gap, we've developed AURORA, a software solution crafted to make the power of statistical and machine learning models more accessible to everyone.

AURORA is designed with the principle that tools that are capable of addressing a diverse range of problems should be within reach of anyone interested in applying scientific methods to their decision-making processes. Our aim is to remove the barriers posed by the need for specialized training, making it easier for individuals to leverage these models in their activities.

Install

Clone this repository or Download zip with project files

CLI:
pip install -r requirements.txt

Launch Aurora

CLI:
python -m Aurora

The process commences with your .csv file containing the requisite information, which is initially imported as a dataframe into AURORA. Subsequently, all models are applied based on this dataframe.

Structuring the Dataframe for plugins

Every plugin comes with its own documentation except the core plugins which are described here.

Regression Algorithms

Within the dataframe, all columns except the last one function as features, while the final column represents the predicted variable. The Linear Regression algorithm can accommodate any type of numerical data in the predicted column, whereas Logistic Regression and Decision Trees are suitable for categorical data.

Mann-Whitney U Test

This test is conducted between two consecutive columns in the dataframe. For instance, if there are four columns named data_1, data_2, data_3, and data_4, the Mann-Whitney U Test is performed between data_1 and data_2, and then between data_3 and data_4, respectively. Consequently, the dataframe must have an even number of columns.

ANOVA

Firs column of the dataframe must contain your tests categories. All other column must be numeric and represents the results of your tests. If your dataframe contains cells without values, AURORA will clean it automatically.

For a practical example, let's consider a scenario where a researcher wants to analyze the impact of three different types of fertilizer on the growth of plants. The researcher has three groups of plants, each group receiving a different type of fertilizer. The goal is to see if there's a significant difference in the growth of plants (measured in height) across these groups.

CSV example:
  Fertilizer_Type  Height_After_1_Month  Height_After_2_Months  Height_After_3_Months
0         Type_A                   5.1                     7.2                     9.8
1         Type_B                   4.8                     7.0                    10.1
2         Type_C                   5.3                     7.9                    10.5
3         Type_A                   5.5                     7.5                     9.9
4         Type_B                   4.9                     7.1                    10.0
5         Type_C                   5.0                     7.8                    10.2
...


Outliers (Anomaly) Detection

This plugin uses Isolation Forest algorithm to detect outliers in timeseries. From your dataframe select column on which you want to apply algorithm. The result will be a plot with both inliers(red) and outliers (blue).

Principal Component Analysis (PCA)

To apply this plugin on your dataframe, the last column must be the target column and others columns must be features columns. The output will be a .csv file with components.


Plugin Development

AURORA supports expanding its functionality through plugins. Users can develop their plugins, add them to the /plugins folder, and AURORA will recognize them. In the current version, AURORA supports two locations on the GUI for placing buttons: one in the Statistics menu item and another in the Machine Learning menu item.

The plugins folder contains all free plugins along with one example plugin for further development.
Please check example_plugin_a.py for instructions regarding plugin development.

Everyone is encouraged to contribute with plugins as this will help AURORA to reach its mission, to become the most comprehensive collection of Statistics and Machine Learning tools available for everyone, for free.


For publishing a plugin please check https://mariusneagoe.com/plugins/
