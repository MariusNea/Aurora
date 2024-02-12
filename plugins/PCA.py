#plugins/pca.py

#####################################################
#### Package: Aurora
#### Plugin: Principal Component Analysis
#### Version: 0.1
#### Author: Marius Neagoe
#### Website: https://mariusneagoe.com
#### Github: https://github.com/MariusNea/Aurora
#####################################################

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def register(app):
    @app.register_plugin('machine_learning','perform_pca_and_export_csv', 'Principal Component Analysis')
    def perform_pca_and_export_csv():
    # Assuming 'df' is a known dataframe available in the scope
        df = app.get_dataframe()
    
    # Separate features from the target
        features = df.columns[:-1]  # Exclude the last column which is the target
        target_column = df.columns[-1]  # The last column is the target
        X = df.loc[:, features].values
        y = df.loc[:, target_column].values
    
    # Standardize the features
        X = StandardScaler().fit_transform(X)
    
    # Perform PCA
        pca = PCA(n_components=2)  # Adjust n_components as needed
        principalComponents = pca.fit_transform(X)
    
    # Create a DataFrame with the principal components
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    
    # Add the target column to the DataFrame
        principalDf[target_column] = y
    
    # Export to CSV
        principalDf.to_csv('pca.csv', index=False)

# introduce in new menu, Machine Learning
