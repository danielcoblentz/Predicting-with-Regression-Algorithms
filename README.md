# Predicting with Regression Algorithms


## What is regression?

Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). In simple terms the regression can be defined as, “Using the relationship between variables to find the best fit line or the regression equation that can be used to make predictions.

This technique is commonly used for forecasting, time series modeling, and identifying causal relationships between variables. Several regression techniques are available to make predictions, and these techniques are primarily influenced by three factors: the number of independent variables, the type of dependent variable, and the shape of the regression line.

![Regression model](/images/Regression.png)
<p align="center">Figure 1: Regression Model</p>


## Project Description


This project implements and compares regression algorithms to predict housing prices based on features like location, size, and property attributes. Using Python and scikit-learn, it explores Linear Regression, Polynomial Regression, Decision Trees, and Random Forests, evaluating their performance with metrics like Mean Squared Error (MSE) and R².

The goal is to identify the most effective model for accurate predictions while highlighting the strengths and limitations of each. Data preprocessing, including handling missing values, feature scaling, and dataset splitting, ensures optimal model performance.


## 1. Simple Linear Regression**

The goal of a linear regression model is to establish a relationship between one or more independent variables (features) and a continuous dependent variable (target). When there is only one independent variable, it is referred to as Univariate Linear Regression, whereas the presence of multiple independent variables is known as Multiple Linear Regression.

<p align="center"> y = w₀ + w₁⋅x₁ </p>
- y: The predicted value or dependent variable. This is the target value the model aims to predict.
w
0
w 
0
​	
 : The intercept or bias term. It represents the value of 
y
y when all other independent variables (
x
1
x 
1
​	
 , 
x
2
x 
2
​	
 , etc.) are zero.
w
1
w 
1
​	
 : The weight or coefficient for 
x
1
x 
1
​	
 . It indicates the strength and direction of the relationship between 
x
1
x 
1
​	
  and 
y
y. A positive 
w
1
w 
1
​	
  means 
y
y increases as 
x
1
x 
1
​	
  increases, while a negative 
w
1
w 
1
​	
 means 
y
y decreases as 
x
1
x 
1
​	
  increases.
x
1
x 
1
​	
 : The independent variable or input feature. This is the input value used to predict 
y
y.


 


## 2. Multiple Linear Regression
![multiple linear regression](https://cdn.xlstat.com/media/feature/0001/03/thumb_2138_feature_medium.png)


## 3. Polynomial Regression

polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x.

<p align="center"> f( x ) = c0 + c1 x + c2 x2 ⋯ cn xn </p>
 n is the degree of the polynomial, and c is a set of coefficients.


## 4. Decision Tree Regression




## 5. Random Forrest Regression


 ## Dataset information
- Dataset: [Housing Prices]()
- Training size: 20,000 images
- Validation size: 6,000 images
- Test size: 14,000 images
- Total size: 2 GB
- GPU: Google Colab A100





## Model evaluation
Model evaluation leads a Data Scientist in the right direction to select or tune an appropriate model.There are three main errors (metrics) used to evaluate regression models, Mean absolute error, Mean Squared error and R2 score.

**Mean Absolute Error (MAE)** tells us the average error in units of y, the predicted feature. A value of 0 indicates a perfect fit.

**Root Mean Square Error (RMSE)** indicates the average error in units of y, the predicted feature, but penalizes larger errors more severely than MAE. A value of 0 indicates a perfect fit.

**R-squared (R2 )** tells us the degree to which the model explains the variance in the data. In other words how much better it is than just predicting the mean.

- A value of 1 indicates a perfect fit.
- A value of 0 indicates a model no better than the mean.
- A value less than 0 indicates a model worse than just predicting the mean.