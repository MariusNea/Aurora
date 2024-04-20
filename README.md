<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/MariusNea/Aurora">
    <img src="images/logo.png" alt="Logo" width="128" height="128">
  </a>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
  <h3 align="center">Aurora</h3>

  <p align="center">
    Problem solving focused statistical and machine learning software toolkit.
    <br />
    <a href="https://github.com/MariusNea/Aurora/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/MariusNea/Aurora/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

In today's world, the fields of statistics and machine learning hold immense potential for solving real-world problems and significantly impacting businesses and daily life. However, the complexity and learning curve associated with these fields can be daunting, making it challenging for those interested to effectively utilize these tools. Recognizing this gap, we've developed AURORA, a software solution crafted to make the power of statistical and machine learning models more accessible to everyone.

AURORA is designed with the principle that tools that are capable of addressing a diverse range of problems should be within reach of anyone interested in applying scientific methods to their decision-making processes. Our aim is to remove the barriers posed by the need for specialized training, making it easier for individuals to leverage these models in their activities.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* ![Matplotlib][Matplotlib]
* ![Pandas][Pandas]
* ![Scikit-learn][scikit-learn]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Prerequisites

Make sure you have Python >=3.9 installed

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MariusNea/Aurora.git
   ```
2. Install libraries
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

   ```sh
   python -m Aurora
   ```
The process commences with your .csv file containing the requisite information, which is initially imported as a dataframe into AURORA. Subsequently, all models are applied based on this dataframe.

<h4>Structuring the Dataframe for plugins</h4>

Every plugin comes with its own documentation except the core plugins which are described here.

<h5>Regression Algorithms</h5>

Within the dataframe, all columns except the last one function as features, while the final column represents the predicted variable. The Linear Regression algorithm can accommodate any type of numerical data in the predicted column, whereas Logistic Regression and Decision Trees are suitable for categorical data.

<h5>Mann-Whitney U Test</h5>

This test is conducted between two consecutive columns in the dataframe. For instance, if there are four columns named data_1, data_2, data_3, and data_4, the Mann-Whitney U Test is performed between data_1 and data_2, and then between data_3 and data_4, respectively. Consequently, the dataframe must have an even number of columns.

<h5>ANOVA</h5>

Firs column of the dataframe must contain your tests categories. All other column must be numeric and represents the results of your tests. If your dataframe contains cells without values, AURORA will clean it automatically.

For a practical example, let's consider a scenario where a researcher wants to analyze the impact of three different types of fertilizer on the growth of plants. The researcher has three groups of plants, each group receiving a different type of fertilizer. The goal is to see if there's a significant difference in the growth of plants (measured in height) across these groups.

CSV example:
 |No   |     Fertilizer_Type | Height_After_1_Month | Height_After_2_Months | Height_After_3_Months |
 |-----|---------------------|---------------------|------------------------|-----------------------|
| 0    |     Type_A          |         5.1         |            7.2         |           9.8 |
| 1    |     Type_B          |         4.8         |            7.0         |          10.1 |
| 2    |     Type_C          |         5.3         |            7.9         |          10.5 |
| 3    |     Type_A          |         5.5         |            7.5         |           9.9 |
| 4    |     Type_B          |         4.9         |            7.1         |          10.0 |
| 5    |     Type_C          |         5.0         |            7.8         |          10.2 |
...


<h5>Outliers (Anomaly) Detection</h5>

This plugin uses Isolation Forest algorithm to detect outliers in timeseries. From your dataframe select column on which you want to apply algorithm. The result will be a plot with both inliers(red) and outliers (blue).

<h5>Principal Component Analysis (PCA)</h5>

To apply this plugin on your dataframe, the last column must be the target column and others columns must be features columns. The output will be a .csv file with components.

<h4>Screenshots from main GUI</h4>

![Product Name Screen Shot][product-screenshot]
![Product Name Screen Shot2][product-screenshot2]
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] Implement Plot & Brush
- [x] Implement Dataframe Edit 
- [x] Implement Dataframe Pagination for fast loading
- [x] Implement Linear Regression
- [x] Implement Logistic Regression
- [x] Implement Decision Tree
- [x] Implement Time Series Decomposition
- [x] Implement One Way ANOVA
- [x] Implement Canonical Correlation Analysis
- [x] Implement Exponential Smoothing Model
- [x] Implement Mann-Whitney U Test
- [x] Implement Poisson Probabilities
- [x] Implement Anomaly (Outliers) Detection
- [x] Implement Principal Component Analysis
- [ ] Implement Automated Model Selector 
- [ ] Implement Support Vector Machines


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

For contributing to the project follow steps described <a href="https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project">here</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GPL-2.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Find more <a href="https://mariusneagoe.com">here</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[product-screenshot]: images/ss1.png
[product-screenshot2]: images/ss2.png
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
