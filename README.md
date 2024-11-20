# Prediction with Regression Algorithms


## What is regression?

Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). In simple terms the regression can be defined as, “Using the relationship between variables to find the best fit line or the regression equation that can be used to make predictions."

This technique is commonly used for forecasting, time series modeling, and identifying causal relationships between variables. Several regression techniques are available to make predictions, and these techniques are primarily influenced by three factors: the number of independent variables, the type of dependent variable, and the shape of the regression line.

![Regression model](/images/Regression.png)
<p align="center">Figure 1: Regression Model</p>


## Project Description


This project implements and compares regression algorithms to predict housing prices based on features such as location, size, and property attributes. Using Python and scikit-learn, the project explores multiple models:

- Linear Regression
- Polynomial Regression
- Decision Trees
- Random Forests

These models are evaluated using metrics like Mean Squared Error (MSE) and R² to assess performance and accuracy. Data preprocessing steps such as handling missing values, feature scaling, and splitting datasets ensure the models are trained and tested optimally.


## 1. Simple Linear Regression

The goal of a linear regression model is to establish a relationship between one or more independent variables (features) and a continuous dependent variable (target). When there is only one independent variable, it is referred to as Univariate Linear Regression, whereas the presence of multiple independent variables is known as Multiple Linear Regression.

<p align="center" style="font-size: 18px;"> General equation: y = w₀ + w₁⋅x₁</p>



## 2. Multiple Linear Regression

Multiple linear regression is used to estimate the relationship between two or more independent variables and one dependent variable. This statistical method models the dependent variable as a linear combination of the independent variables and is widely used to predict outcomes and analyze relationships in various domains.
<div align="center">

![multiple linear regression](/images/multi_linear_regression2.png)

</div>
<p align="center">Figure 2: Multiple Linear Regression Model</p>



## 3. Polynomial Regression

Polynomial Regression is an extension of Linear Regression that models the relationship between the dependent and independent variables as an 
𝑛
n-degree polynomial. It is useful when the data exhibits a non-linear relationship that cannot be captured by a straight line.

Unlike simple linear regression, which assumes a linear relationship between the variables, polynomial regression fits a curvilinear equation.

![Polynomial_regression](/images/polynomial%20regression.png)
<p align="center">Figure 3: Polynomial Linear Regression Model</p>


## 4. Decision Tree Regression
A Decision Tree is a flowchart-like structure used in machine learning for both classification and regression tasks. Each node in the tree represents a decision or a test on a specific feature, directing the data to subsequent nodes based on the outcome of the test. The tree structure continues to split the data into smaller subsets until it reaches a terminal node, known as a leaf, which provides the final prediction as an output.

In a Decision Tree:

- Root Node: Represents the initial dataset and contains the first decision or test to split the data.
- Internal Nodes: Represent decisions or conditions based on one feature. Each internal node splits the data into two or more branches.
- Leaf Nodes: Represent the outcome or prediction (e.g., a class label for classification or a numeric value for regression).

![Decision_tree_img](/images/decision%20tree.png)
<p align="center">Figure 4: Decision Tree Regression Model</p>


## 5. Random Forrest Regression
Random Forest is an ensemble learning algorithm used for classification and regression tasks. It operates by constructing multiple decision trees during training and combines their outputs (majority vote for classification or average for regression) to produce a more accurate and robust prediction. By aggregating the results of many decision trees, Random Forest reduces the risk of overfitting and increases the model's generalization ability.


![Random_Forrest_Regression](/images/random_forrest.png)
<p align="center">Figure 5: Random Forrest Regression Model</p>


## Model evaluation
There are three main errors (metrics) used to evaluate regression models, Mean absolute error, Mean Squared error and R2 score.

**Mean Absolute Error (MAE)** The average magnitude of prediction errors in the same unit as the dependent variable.

- 0 indicates a perfect fit.

**Root Mean Square Error (RMSE)** indicates the average error in units of y, the predicted feature, but penalizes larger errors more severely than MAE. 

- A value of 0 indicates a perfect fit.

**R-squared (R2 )** tells us the degree to which the model explains the variance in the data. In other words how much better it is than just predicting the mean.

- 1: Perfect fit.
- 0: Model explains no variance.
- < 0: Model performs worse than simply predicting the mean


## Model results
### Comparative Performance
| Model                     | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R² Score  |
|---------------------------|---------------------------|---------------------------|----------|
| Simple Linear Regression  | 0                         | 0                         | 0        |
| Multiple Linear Regression| 0                         | 0                         | 0        |
| Polynomial Regression     | 0                         | 0                         | 0        |
| Decision Tree Regression  | 0                         | 0                         | 0        |
| Random Forest Regression  | 0                         | 0                         | 0        |

### **Observations**
- **Random Forest Regression** performed the best, achieving the lowest MSE and the highest R² score.
- **Simple Linear Regression** showed the weakest performance due to its inability to capture non-linear relationships.

<p align="center">
  <img src="/images/model_performance.png" alt="Model Performance Comparison">
</p>
<p align="center">Figure 6: Model Performance Comparison</p>

## References
