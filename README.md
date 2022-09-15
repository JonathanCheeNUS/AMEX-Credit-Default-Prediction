Project Description:
Credit cards are increasingly being used in almost all aspects of our life, from shopping on Amazon to dining at a restaurant. For credit card issuers, knowing whether a customer will pay back what he spent is an important and challenging problem to address. Credit default prediction is critical to risk management for card issuers. It allows lenders to optimise lending decisions and detect red flags in advance for better customer experience and business outcome. 

In this project we will seek  to predict the probability that an American Express customer does not pay back their credit card balance amount in the future based on their monthly customer profile. We will apply various machine learning techniques on the industrial scale dataset to more accurately predict customers’ credit default rates based on their lending behaviour in the previous time points. 
The project would be executed in the following stages. Our team will first conduct exploratory data analysis, including data visualisation, to better understand the nature of the data. Feature engineering and feature selection will then be performed to create more useful features with higher predictive capability. Lastly, we will experiment with different machine learning algorithms and perform model selection using cross validation. 

Source of Data:
The data comes from a Kaggle competition. There are 3 available csv files, `train_data.csv`, `test_data.csv` and `train_labvels.csv`. In the training data, there are 5,531,451 records in total, with 458,913 unique customer IDs. The testing data contains 11,363,762 records in total with 924,621 unique customer IDs. 

The target binary variable is derived by observing the customer across a 18 months performance window after the latest credit card statement. If the customer does not pay the due amount in 120 days after their latest statement date, this would be considered a default event (represented as 0 in the target column).
The dataset contains aggregated profile features for each customer at each statement date. There are 188 features in the data. Features are anonymized and normalised, and fall into the following general categories:
D_* = Delinquency variables
S_* = Spend variables
P_* = Payment variables
B_* = Balance variables
R_* = Risk variables

For more details, please refer to the website: https://www.kaggle.com/competitions/amex-default-prediction/data. 

Machine Learning Algorithms:
This project deals with a classification problem. Hence, we will use some basic classification models such as SVM, Logistic Regression as our baseline models, followed by several more sophisticated and advanced classification models such as random forest, LightGBM and XGBoost. 

In addition, deep learning models will be used, including Deep Neural Networks, LSTM and Multi-Layer Perceptron Bagging. In particular, LSTM will be used twice for different purposes, first to generate embeddings from the time-series data during feature engineering, and second to make predictions directly.

Finally, after getting the predictions from various models, we will use an ensemble method to merge all results to get the final prediction.

Evaluation Metric:
The evaluation metric, 𝑀, for this project is the mean of two measures of rank ordering: Normalised Gini Coefficient, 𝐺, and default rate captured at 4%, 𝐷.
𝑀=0.5⋅(𝐺+𝐷)
The default rate captured at 4% is the percentage of the positive labels (defaults) captured within the highest-ranked 4% of the predictions, and represents a Sensitivity/Recall statistic.
For both of the sub-metrics 𝐺 and 𝐷, the negative labels are given a weight of 20 to adjust for downsampling.
This metric has a maximum value of 1.0.
