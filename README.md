# NBA Most-Improved-Player Prediction Model
ML Model to predict the Most-Improved-Player win-shares received by a given NBA player

#### -- Project Status: Completed

## Project Description

* Created a tool that estimates the MIP win-shares received by an NBA player (r<sup>2</sup> ~ 0.165)
* Engineered new dataset that quantifies changes in player performance between consecutive seasons (in order to gauge improvement)
* Performed recursive feature selection to separate most relevant features
* Optimized Ridge Regression, Lasso Regression, and kNN Regression using GridsearchCV to reach the best model.

## Purpose/Objective

The purpose of this project is to build a Regression Machine Learning model that can estimate the MIP win-shares of
an NBA player in their current season, given data of their change in performance from the previous season. The target 
applications of this model are widespread: we are essentially trying to predict the next breakout star. Quantifying the 
likelihood and projection for a player’s improvement plays a factor in the free agent market of every team. A team can 
assess the risk of signing a particular player to a contract in hopes that they will play better than anticipated, 
making their contract a great value. Projecting who will be the next breakout player also plays a factor in sports 
gambling. If your model projects that a certain player is on an upwards trend and you believe that sportsbooks have them
undervalued, you can leverage this information. Furthermore, sportswear companies such as Nike and Adidas can use this 
information to identify and sign the next up-and-coming star player in order to market their products with the player's 
name and likeness, which can potentially generate significant profits.

## Code and Resources Used

**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, joblib, pickle  

[comment]: <> (**For Web Framework Requirements:**  ```pip install -r requirements.txt```  )
**Raw Dataset:** https://www.basketball-reference.com/leagues/NBA_2021_per_game.html  
**Raw Dataset:** https://www.basketball-reference.com/awards/mip.html  
**Raw Dataset:** https://www.basketball-reference.com/awards/awards_2021.html#mip

## Methods Used

* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling

## Data Collection and Cleaning

Player and award data was acquired from [Basketball Reference](https://www.basketball-reference.com/).
\
\
After downloading the data, it required cleaning to be usable for our model. The following changes were made:

* Filled NA values with 0
* Created a list of unique player names
* Created a list of numerical features in the dataset
* Used custom functions to create a new dataset of performance changes between subsequent seasons
* Inserted continuous MIP win-shares values for each corresponding player entry
* Resulted with 7606 samples and 27 features.

## EDA

We examined the distributions of the data and the value counts for the various quantitative and categorical variables.
Below are a few highlights from the analysis.

![alt text](https://github.com/jjz17/NBA-MIP-Prediction-Model/blob/main/visualizations/histogram.png "Distribution of Changes in Points Scored")
![alt text](https://github.com/jjz17/NBA-MIP-Prediction-Model/blob/main/visualizations/pairplots.png "Various Stats against MIP win-shares")
![alt text](https://github.com/jjz17/NBA-MIP-Prediction-Model/blob/main/visualizations/pca_scatter.png "PCA Dimensionality Reduction")

## Model Building

First, we split the data into train and tests sets with a test size of 25%.

We tried three different models and evaluated them using the coefficient of determination (r<sup>2</sup>). We chose 
r<sup>2</sup> because it is relatively easy to interpret and it can also guide us to avoid overfitting the model.

We also performed recursive feature elimination to select the most relevant features

Finally, we used GridSearchCV to perform hyperparameters tuning

We used three different models:

* **Ridge Regression** – Prevents overfitting with L2 regularization
* **Lasso Regression** - Performs automatic feature selection with L1 regularization.
* **kNN Regressor** – Intuitive algorithm and makes no assumptions about the data and its distribution.

## Model performance

After tuning, the kNN Regressor performed significantly better on the test and validation sets.

* **Ridge Regression**: r<sup>2</sup> = 0.050, MSE = 0.00178
* **Lasso Regression**: r<sup>2</sup> = -3.608e-05, MSE = 0.00187
* **kNN Regressor**: r<sup>2</sup> = 0.165, MSE = 0.00156

## Conclusion

We conclude that there is likely a non-linear relationship between the faetures and the target. The kNN Regressor shows 
promise and could be combined with additional features (advanced stats, etc.) to yield more accurate predictions. 
Furthermore, we can apply more advanced ML algorithms such as neural networks and deep learning to achieve greater
performance.

## Contributors

* Kody Yuen @**yuen.k@northeastern.edu**
* Corey An @**an.co@northeastern.edu**

## Project Structure

- raw dataset and preprocessed dataset are included under
  the [data](https://github.com/jjz17/NBA-MIP-Prediction-Model/tree/main/data) directory
- model and scaler objects are included under
  the [models](https://github.com/jjz17/NBA-MIP-Prediction-Model/tree/main/models) directory
- Jupyter Notebook work is included under
  the [notebooks](https://github.com/jjz17/NBA-MIP-Prediction-Model/tree/main/notebooks) directory
- produced presentation is included under the [presentation](https://github.com/jjz17/NBA-MIP-Prediction-Model/tree/main/presentation) directory
- all source code (data wrangling, exploratory data analysis, model building, custom functions) is included under
  the [src](https://github.com/jjz17/NBA-MIP-Prediction-Model/tree/main/src) directory
- produced visualizations are included under
  the [visualizations](https://github.com/jjz17/NBA-MIP-Prediction-Model/tree/main/visualizations) directory
