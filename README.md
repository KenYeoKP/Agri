# A Machine Learning Project
## **Regression** - _predicting temperature conditions_
## **Classification** - _classifying Plant Type and Stages_

## 1. Overview of the Submitted Folder and Folder Structure
This repository contains the necessary files and scripts to execute an end-to-end machine learning pipeline for predicting temperature conditions and classifying plant types and stages based on data from sensors.

## 2. Flow of pipeline
1. Data loading and understanding
2. Conduct Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Model development
5. Hyperparameter tuning
6. Model selection

## 3. Overview of key findings in EDA
1. The original dataset comprised 57,489 records.
2. The 4 categorical data fields exhibited an even distribution of values, as confirmed through visual analysis.
3. The charts for the 11 numerical fields revealed the following:

| Feature          | Description                                         |
|------------------|-----------------------------------------------------|
| Temperature      | Shows a moderate normal distribution centered around 23.3 degrees Celsius. |
| Humidity         | Displays a bimodal distribution with peaks around 64% and 74%. |
| Light Intensity  | Data is spread over a wide range with majority between 150 lux and 800 lux. |
| co2              | Data is spread over a wide range between 800 ppm and 1500 ppm. It has a multimodal distribution with many peaks. |
| ec Sensor        | Values fall between 0 ds/m and 2 ds/m with a majority at 1 ds/m. |
| o2               | Majority of the data falls between 4 ppm and 10 ppm, with peaks at 6 ppm and 7 ppm. |
| n Sensor         | Moderate distribution but data is spread over a wide range between 50 ppm and 250 ppm. |
| p Sensor         | Data is evenly distributed with mean is close to median values around 50 ppm. |
| k Sensor         | Data is slightly skewed to the right, which suggests more data is having values greater than 215 ppm (mean). |
| ph Sensor        | Majority of the ph values is approximately at 6. |
| Water Level      | Displays a multimodal distribution with majority of the values falling between 20 mm and 35 mm. | 

4. Bivariate analysis reviewed correlation between temperature and the following environmental factors:

| Feature         | Description                                         |
|-----------------|-----------------------------------------------------|
| Humidity        | High humidity can trap heat and potentially raise temperature | 
| Light Intensity | Light contributes to heating |
| co2             | A greenhouse gas which contributes to warming |
| Water Level     | Can be due to rainfall or irrigation; both can lower the temperature |

5. The correlation matrix indicated a moderate correlation between temperature and environmental factors such as humidity (0.28), light intensity (0.25), and CO2 levels (0.16). These factors can be used as input features for temperature prediction. Water level, with a correlation coefficient of -0.06, appears to have minimal predictive power for temperature.
6. A thorough data cleaning process identified and removed 7,489 duplicate records.
7. The dataset, comprising data from eleven sensors, will be used to categorize plants based on their type and stage of growth. The correlation analysis revealed relationships between Plant Type-Stage and various sensor readings:

| Feature         | Description                                         |
|-----------------|-----------------------------------------------------|
| co2             | Strong negative correlation (-0.59) |
| Light Intensity | Moderate negative correlation (-0.46) |
| ph Sensor       | Moderate negative correlation (-0.45) |
| k Sensor        | Moderate negative correlation (-0.33) |
| Temperature     | Weak negative correlation (-0.28) |
| o2 Sensor       | Weak negative correlation (-0.25) |
| n Sensor        | Weak negative correlation (-0.22) |
| p Sensor        | Weak negative correlation (-0.21) |
| Humidity        | Weak positive correlation (0.14)  |
| ec Sensor       | Weak negative correlation (-0.08) |
| Water Level     | Weak positive correlation (0.05)  |

## 4. Features processing

| S/N | Feature          | Description                                                    |
|-----|------------------|----------------------------------------------------------------|
| 1.  | System Location  | Converted to lowercase, categorical data type.                 |
| 2.  | Previous Cycle   | Converted to lowercase, categorical data type.                 |
| 3.  | Plant Type       | Converted to lowercase, categorical data type.                 |
| 4.  | Plant Stage      | Converted to lowercase, categorical data type.                 |
| 5.  | Temperature      | Imputed median values onto null data. Convert negative to positive values using .abs() function. |
| 6.  | Humidity         | No imputation. A subset of working data will be created based on humidity records.               |
| 7.  | Light Intensity  | Imputed median values onto null data. Convert negative to positive values using .abs() function. |
| 8.  | CO2              | No action.                                                     |
| 9.  | EC Sensor        | Convert negative to positive values using .abs() function.     |
| 10. | O2               | No action.                                                     |
| 11. | N Sensor         | Imputed median values onto null data. Remove 'ppm' from data.  |
| 12. | P Sensor         | Imputed median values onto null data. Remove 'ppm' from data.  |
| 13. | K Sensor         | Imputed median values onto null data. Remove 'ppm' from data.  |
| 14. | pH Sensor        | No action.                                                     |
| 15. | Water Level      | Imputed median values onto null data.                          |
| 16. | Plant Type-Stage | Concatenate Plant Type and Plant Stage, and assigned as it categorical data type.                |

## 5. Explanation on choice of models
### Objective 1 - Predict Temperature Conditions
The following models were evaluated to predict temperature conditions:  
<br>Linear Regression:  
A fundamental statistical method that models the relationship between independent variables and a dependent variable by fitting a linear equation. It is simple and interpretable, making it a good baseline model.<br>
<br>Ridge Regression:  
An extension of linear regression that includes L2 regularisation to prevent overfitting by penalising large coefficients. This model is useful when dealing with multicollinearity or when the number of predictors is high relative to the number of observations.<br>
<br>Lasso Regression:  
Similar to Ridge regression but uses L1 regularisation, which can shrink some coefficients to zero, effectively performing variable selection. This is beneficial for models where feature selection is important.<br>
<br>Decision Tree Regressor:  
A non-linear model that splits the data into subsets based on feature values, creating a tree-like structure. It is easy to interpret and can capture complex relationships but may overfit if not properly tuned.<br>
<br>Random Forest Regressor:  
An ensemble method that combines multiple decision trees to improve predictive accuracy and control overfitting. It averages the predictions from individual trees, making it robust against noise in the dataset.<br>
<br>Support Vector Regressor (SVR):  
A regression technique based on Support Vector Machines, which tries to find a function that deviates from the actual target values by a value no greater than a specified margin. SVR is effective in high-dimensional spaces and works well with non-linear relationships.<br>
<br>Gradient Boosting Regressor:  
Another ensemble method that builds trees sequentially, where each new tree attempts to correct errors made by previously trained trees. Gradient boosting can provide high predictive accuracy and is particularly effective for complex datasets.<br>


| Model            	         | MAE  |  MSE | RMSE	| R² Score	| Runtime (seconds) |
|-----------------------------|------|------|------|-----------|-------------------|
| Linear Regression	         | 1.01 |	1.68 | 1.30 | 0.24      | 4.07              |
| Ridge Regression      	   | 1.01 | 1.68 | 1.30 | 0.24      | 2.76              |
| Lasso Regression            | 1.10 | 2.21 | 1.49	| -0.00     | 1.76              |
| Decision Tree Regressor   	| 1.22 | 2.63 | 1.62	| -0.19     | 0.23              |
| Random Forest Regressor	   | 0.92 | 1.44 | 1.20 | 0.35      | 5.97              |
| SVR	                        | 0.89 | 1.44 | 1.20 | 0.35      | 8.82              |
| Gradient Boosting Regressor | 0.88 | 1.29 | 1.14 | 0.42      | 1.33              |

### Objective 2 - Categorise Plant Type-Stage
The following models were evaluated to categorise Plant Type-Stage:  
<br>Decision Tree Classifier:  
The Decision Tree Classifier is a non-parametric supervised learning algorithm used for classification and regression tasks. It works by splitting the dataset into subsets based on the value of input features, creating a tree-like model of decisions. This model is easy to interpret and visualise, making it suitable for understanding feature importance and decision-making processes.<br>
<br>Random Forest Classifier:  
Random Forest is an ensemble learning method that constructs multiple decision trees during training and merges their outputs to improve accuracy and control overfitting. By averaging the results of many trees, it reduces the variance associated with individual decision trees, making it robust against noise in the dataset.<br>
<br>Logistic Regression:  
Logistic Regression is a statistical method used for binary classification problems. It estimates the probability that a given instance belongs to a particular class using a logistic function. Despite its name, it is a linear model that works well when the relationship between the features and the target variable is approximately linear.<br>
<br>K-Neighbors Classifier:  
The K-Neighbors Classifier is a simple, instance-based learning algorithm that classifies data points based on their proximity to other instances in the feature space. It relies on the majority class among the 'k' nearest neighbours to make predictions.<br>
<br>Support Vector Classifier (SVC):  
The Support Vector Classifier is a supervised learning algorithm that aims to find the optimal hyperplane that separates different classes in high-dimensional space. It works well for both linear and non-linear classification tasks by using kernel functions to transform data into higher dimensions, allowing it to handle complex relationships between features.<br>
<br>Gradient Boosting Classifier:  
Gradient Boosting is an ensemble technique that builds models sequentially, where each new model attempts to correct the errors made by previous ones. By optimising a loss function through gradient descent, it creates strong predictive models that can handle various types of data distributions effectively.<br>

| Model	                    | Accuracy	   | Precision	    | Recall	     | F1 Score	   |Runtime (seconds) |
|----------------------------|---------------|---------------|---------------|---------------|------------------|
| DecisionTreeClassifier	  | 0.79 +/- 0.01	| 0.79 +/- 0.01 | 0.79 +/- 0.01 | 0.79 +/- 0.01 | 3.63             |
| RandomForestClassifier	  | 0.81 +/- 0.00	| 0.82 +/- 0.00 | 0.81 +/- 0.00 | 0.81 +/- 0.00	| 4.97             |
| LogisticRegression	        | 0.78 +/- 0.00	| 0.78 +/- 0.00 | 0.78 +/- 0.00 | 0.78 +/- 0.00 | 2.77             |
| KNeighborsClassifier	     | 0.76 +/- 0.01	| 0.76 +/- 0.01 | 0.76 +/- 0.01 | 0.76 +/- 0.01 | 0.65             |
| SVC                  	     | 0.80 +/- 0.00	| 0.81 +/- 0.00 | 0.80 +/- 0.00 | 0.80 +/- 0.00	| 2.99             |
| GradientBoostingClassifier | 0.81 +/- 0.01	| 0.81 +/- 0.01 | 0.81 +/- 0.01 | 0.81 +/- 0.01 | 29.06            |

## 6. Evaluation of models developed
### Objective 1 - Predict Temperature Conditions
The Gradient Boosting model exhibited the strongest predictive performance, achieving the lowest Mean Absolute Error (MAE) of 0.88, Mean Squared Error (MSE) of 1.29, and Root Mean Squared Error (RMSE) of 1.14. This model also demonstrated a reasonable computational efficiency.

The Ridge Regression model, while not as powerful as Gradient Boosting, still produced respectable results with a MAE of 1.01, MSE of 1.68, and RMSE of 1.30. Its simpler structure compared to ensemble methods makes it a suitable option for this task where computational efficiency is a priority due to the deadline.

Training on X_train data sets were carried out after hyperparameter tuning. Both models were subsequently evaluated using validation dataset and the results are as follows:

| Model                   | Mean Absolute Error (MAE) | Mean Squared Error (MSE)  | Root Mean Squared Error (RMSE)  | R-squared (R²)  |
|-------------------------|---------------------------|---------------------------|---------------------------------|-----------------|
| Ridge Regression        | 1.02                      | 1.67                      | 1.29                            | 0.24            |
| Gradient Boosting       | 0.87                      | 1.25                      | 1.12                            | 0.43            |

Lastly, both models were evaluated using testing data and the results are as follows:

| Model                   | Mean Absolute Error (MAE) | Mean Squared Error (MSE)  | Root Mean Squared Error (RMSE)  | R-squared (R²)  |
|-------------------------|---------------------------|---------------------------|---------------------------------|-----------------|
| Ridge                   | 1.01                      | 1.67                      | 1.29                            | 0.22            |
| Gradient Boosting       | 0.87                      | 1.26                      | 1.12                            | 0.41            |

### Result - Objective 1
Gradient Boosting consistently outperformed Ridge Regression across all metrics in both evaluations, therefore, it is the preferred choice due to its superior predictive capabilities.

### Objective 2 - Categorise Plant Type-Stage
The evaluation of various classification models revealed that Random Forest and Gradient Boosting emerged as the top performers, both achieving an accuracy of 81%. While these ensemble methods offer better accuracy, their computation complexity and longer training times make them impractical to meet the deadline.

Considering the constraints of time and computational resources, a decision was made to prioritise models that strike a balance between accuracy and efficiency. The Decision Tree and Logistic Regression models were selected as suitable alternatives.

The Decision Tree classifier demonstrated robust performance, achieving an accuracy of 79%. Logistic Regression, with an accuracy of 78%, offers a probabilistic approach to classification. It excels in its simplicity and fast execution time, making it well-suited for this task.

In addition to Accuracy, Precision is another key metric as wrongly classifying Plant Type-Stage may result in mismanagement of resources and ultimately affect crop yields. Accurate classification is crucial for ensuring that the right resources are allocated to each plant type at its specific growth stage.

Training on X_train data sets were carried out after hyperparameter tuning. Both models were subsequently evaluated using validation dataset and the results are as follows:

| Model	                    | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| DecisionTreeClassifier	  | 0.80	    | 0.81      | 0.80   | 0.80     |
| LogisticRegression	        | 0.77     | 0.77      | 0.77   | 0.77     |

Lastly, both models were evaluated using testing data and the results are as follows:

| Model	                    | Accuracy | Precision | Recall | F1 Score	|
|----------------------------|----------|-----------|--------|----------|
| DecisionTreeClassifier	  | 0.79     | 0.79      | 0.79   | 0.79     |
| LogisticRegression	        | 0.77     | 0.77      | 0.77   | 0.77     |

The Decision Tree Classifier outperforms Logistic Regression across all metrics on both validation and test datasets. Both models are able to generalise well to new data. In this analysis, the Decision Tree Classifier would be the preferred choice as compared to Logistic Regression. However, it is vital to recognise that despite carrying out hyperparameter tuning, the performance of both models did not increase significantly and they still failed to perform better than Random Forest and Gradient Boosting models.

### Result - Objective 2
For the purpose of this exercise, the Decision Tree model is the preferred model for categorising Plant Type-Stage due to its computational efficiency.

