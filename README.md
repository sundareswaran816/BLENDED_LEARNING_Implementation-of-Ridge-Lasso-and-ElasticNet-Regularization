# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the car price dataset into the program.
2. Preprocess the dataset and generate polynomial features using `PolynomialFeatures`.
3. Build pipelines for Ridge, Lasso, and ElasticNet regression models and train them using the training data.
4. Evaluate and compare the performance of the models using test data and metrics such as Mean Squared Error (MSE) and R² score.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
data= pd.read_csv("encoded_car_data (1).csv")
data.head()
#Data preprocessing 
#data =data.drop(['CarName','car_ID'],axis=1)
data = pd.get_dummies(data, drop_first=True)

#Splitting the data into features and target variable 
X = data.drop('price', axis=1)
y = data['price']

#Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1,1))

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#define the models and pipelines
models ={
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0,l1_ratio=0.5)
}

#Dictionafry to store results 
results={}

#Traom ajd evaluate eac model
for name, model in models.items():
    #Create a pipeline with polynomial features and the model
    pipeline= Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
    
    #Fit the model
    pipeline.fit(X_train, y_train)
    
    #Make predictions
    predictions= pipeline.predict(X_test)
    
    #Calculate performance matrics
    mse= mean_squared_error(y_test, predictions)
    r2= r2_score(y_test, predictions)
    
    #Store results 
    results[name]= {'MSE': mse, 'R² Score': r2}
    
    
#Print results
print('Name: Sundareswaran K')
print('Reg_No:212225040439')
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f},R² Score: {metrics['R² Score']:.2f}")
    
    
#Visualization of the results
#Convert results to DataFrame for easier plotting 
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

#Set the figure size
plt.figure(figsize=(12,5))

#Bar plot for MSE
plt.subplot(1,2,1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)


#Bar plot for R² Score
plt.subplot(1,2,2)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

#Show the plots
plt.tight_layout()
plt.show()
```

## Output:
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/f256aba8-4f01-4cd2-b407-83c988c81465" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
