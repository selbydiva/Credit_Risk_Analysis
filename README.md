# Credit_Risk_Analysis

The goal of this project is to assess whether a customer is a good or bad borrower based on customers historical credit payment data. We have three datasets available for this analysis:

1.	Dataset 1: Female Customer Data
2.	Dataset 2: Male Customer Data
3.	Dataset 3: Credit Data

These datasets will be used to develop a predictive model to classify borrowers.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/73646458-4c29-4b88-a3b7-2a378c058f55)

After checking for duplicates and missing values, the datasets contain:

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/2c0092b7-3e63-48f2-8b90-117b45922481)

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/d8763d07-1a0e-469a-8ffe-c32e9db4ea68)

In the credit data, 4,536 customer IDs were detected don’t have credit. Since we only need customers with credit, these 4,536 rows were deleted.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/c129c8a9-04d7-4da2-a61c-2f89f0478b2c)

To combine these three datasets, first we need to join the female customer data and male customer data together. Then, we merge the resulting dataset with the credit data using the same Id_customer.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/86f3099c-87d3-4fd4-8af0-02ea2a3ec5a4)

The joined data between female and male customers revealed 47 duplicated Id_customer entries (94 rows). Since these duplicates are relatively few and appear to be input errors, as they have different genders and other conflicting information, it is better to delete these duplicated Id_customer rows.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/dc750519-4005-47a1-be98-037379914e18)

After deleting the duplicated Id_customer rows, the dataset now contains 32,910 rows and 19 columns. However, the 'Pekerjaan' column has 10,314 missing values. Given the large number of missing values, it is preferable to predict these missing values using a random forest classifier.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/09777f33-4c54-4947-b070-e5f332493c51)

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/4fe4ec02-9c09-4618-826c-b6da80f287cd)

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/4395eb98-1a54-4aea-bdcf-8f218113bc33)

The accuracy for predicting missing value in  ‘Pekerjaan’ column is 91%.

After cleaning all the data, it's time to build a model for classifying customers into 'good borrower' and 'bad borrower'. Feature selection was performed to identify correlated features. The correlated features identified are: 'JK' (gender), 'KepemilikanProperti' (property ownership), 'Pendapatan' (income), 'TipePendapatan' (income type), 'StatusKeluarga' (marital status), 'TipeRumah' (house type), 'Pekerjaan' (occupation), 'Age', and 'Experience'.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/4ad31715-3b00-4e45-be52-417f86113476)

The data is split into 80% for training and 20% for testing.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/f78856bb-16fa-4ef4-8636-3c81664413dd)

As observed, the proportion of 'Overdue' values is imbalanced between good borrowers and bad borrowers. This class imbalance can lead to underfitting or overfitting issues. To address this, a resampling method called oversampling is applied to the data.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/b914c5ab-ec1c-4bff-8513-9559460a63d4)

After applying oversampling to balance the data proportions, the dataset is now more balanced. Next, a model is built using the Random Forest Classifier. The accuracy of this model is 91%.

![image](https://github.com/selbydiva/Credit_Risk_Analysis/assets/154320650/daa25080-9550-44c0-b189-b38b35983876)











