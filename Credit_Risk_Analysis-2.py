#!/usr/bin/env python
# coding: utf-8

# # **DEFINE OBJECTIVE**

# 1. Segmenting customers to determine eligibility for obtaining a credit

# # Data Collection

# Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# Load Dataset

# In[2]:


# Application History Female
path_female = './Downloads/Final_Project_Dataset/application_history_f.xlsx'
df_female = pd.read_excel(path_female)

# Application History male
path_male = './Downloads/Final_Project_Dataset/application_history_m.xlsx'
df_male = pd.read_excel(path_male)

# Credit History
path_credit = './Downloads/Final_Project_Dataset/credit_history.csv'
df_credit = pd.read_csv(path_credit)


# Data Identification

# In[ ]:


print('\033[1mTABEL FEMALE :\033[0m')
print(df_female.describe())
print(' ')

print('\033[1mTABEL MALE :\033[0m')
print(df_male.describe())
print(' ')

print('\033[1mTABEL CREDIT :\033[0m')
print(df_credit.describe())
print(' ')


# In[ ]:


print('\033[1mTABEL FEMALE :\033[0m', len(df_female))
df_female.info()

print('\033[1mTABEL MALE :\033[0m', len(df_male))
df_male.info()

print('\033[1mTABEL CREDIT :\033[0m', len(df_credit))
df_credit.info()


# Null and Empty Value Identification

# In[3]:


column_female = df_female.columns
column_male = df_male.columns
column_credit = df_credit.columns

print('\033[1mTABEL FEMALE :\033[0m')
print('\033[1mKolom Yang Mengandung Data NaN :\033[0m')
for i in column_female:
  data_nan = df_female[i].isnull().sum()
  if data_nan > 0 :
    print(i, ' : ', data_nan)
print('')
print('\033[1mKolom Yang Mengandung Data Kosong (Empty Value) :\033[0m')
for i in column_female:
  data_empty = df_female[i].eq(' ').sum() + df_female[i].eq('-').sum()
  if data_empty > 0 :
    print(i, ' : ', data_empty)
print(" ")
print(" ")

print('\033[1mTABEL MALE :\033[0m')
print('\033[1mKolom Yang Mengandung Data NaN :\033[0m')
for i in column_male:
  data_nan = df_male[i].isnull().sum()
  if data_nan > 0 :
    print(i, ' : ', data_nan)
print('')
print('\033[1mKolom Yang Mengandung Data Kosong (Empty Value) :\033[0m')
for i in column_male:
  data_empty = df_male[i].eq(' ').sum() + df_male[i].eq('-').sum()
  if data_empty > 0 :
    print(i, ' : ', data_empty)
print(" ")
print(" ")

print('\033[1mTABEL CREDIT :\033[0m')
print('\033[1mKolom Yang Mengandung Data NaN :\033[0m')
for i in column_credit:
  data_nan = df_credit[i].isnull().sum()
  if data_nan > 0 :
    print(i, ' : ', data_nan)
print('')
print('\033[1mKolom Yang Mengandung Data Kosong (Empty Value) :\033[0m')
for i in column_credit:
  data_empty = df_credit[i].eq(' ').sum() + df_credit[i].eq('-').sum()
  if data_empty > 0 :
    print(i, ' : ', data_empty)
print(" ")
print(" ")


# 1. there are 104050 rows of 'Pekerjaan' column in df_female are null
# 2. there are 30193 rows of 'Pekerjaan' column in df_male are null

# Duplicate Row Identification

# In[22]:


data_duplikat_female = df_female.duplicated().sum()
print('\033[1mNumber of Duplicate Row Female Data : \033[0m', data_duplikat_female)

data_duplikat_male = df_male.duplicated().sum()
print('\033[1mNumber of Duplicate Row Male Data : \033[0m', data_duplikat_male)

data_duplikat_credit = df_credit.duplicated().sum()
print('\033[1mNumber of Duplicate Row Credit Data : \033[0m', data_duplikat_credit)


# Deletes Duplicated Data Credit

# In[3]:


ex_credit = df_credit[df_credit['Overdue']=='Tidak memiliki pinjaman'].index.tolist()
df_credit_ex = df_credit.drop(ex_credit) #data without 'Tidak memiliki pinjaman'
df_credit_in = df_credit.iloc[ex_credit] #data with 'Tidak memiliki pinjaman'
df_credit_ex['Overdue'] = df_credit_ex['Overdue'].astype('int64')

print('Number of Rows df_credit : ', len(df_credit))
print('Number of Rows df_credit_ex :', len(df_credit_ex))
print('Number of Rows df_credit_in :' , len(df_credit_in))


# In[4]:


#Get unique value
unique_id_ex = list(np.unique(df_credit_ex['Id_customer']))
unique_id_in = list(np.unique(df_credit_in['Id_customer']))

# Get dupicated id_customer
index_drop_ex = []
for i in unique_id_ex:
  if len(df_credit_ex[df_credit_ex['Id_customer']==i]) > 1:
    index_credit_drop = df_credit_ex[df_credit_ex['Id_customer']==i].sort_values(by='Overdue',ascending=False)[1:].index.tolist()
    index_drop_ex.extend(index_credit_drop)

#id_customer that appear in df_credit_ex and df_credit_in
both_id = []
for i in unique_id_in:
  if i in unique_id_ex:
    both_id.append(i)

#id_customer only in data with 'Tidak memiliki pinjaman'
no_both_id = []
for i in unique_id_in:
  if i not in unique_id_ex:
    no_both_id.append(i)

# Get dropped index from id appear in both of data
index_drop_in = []
for a in both_id:
  index_drop = df_credit_in[df_credit_in['Id_customer']==a].index.tolist()
  index_drop_in.extend(index_drop)

# Get dropped index from id appear only in data with 'Tidak memiliki pinjaman'
index_drop_in_not = []
for a in no_both_id:
  index_drop = df_credit_in[df_credit_in['Id_customer']==a].index.tolist()
  index_drop_in_not.extend(index_drop)

# Drop dulicated rows
#dropped duplicate rows df_credit
df_credit = df_credit.drop(index_drop_ex)
df_credit = df_credit.drop(index_drop_in)
df_credit = df_credit.drop(index_drop_in_not)

print('\033[1mDuplicate rows have been removed from the data credit\033[0m')


# Delete Duplicated Rows Female & Male Data for 18 Columns

# In[5]:


# delete duplicate for 18 columns
df_female = df_female.drop_duplicates(keep='first')
df_male = df_male.drop_duplicates(keep='first')

print('\033[1mDuplicate rows have been removed from the data customer female and male\033[0m')


# ## Data Transformation

# Join and Merge Data

# In[6]:


# Joining female and male data
df_join = pd.concat([df_female, df_male],ignore_index=True)

# Delete duplicated rows based on 'Id_customer'
index_to_drop = df_join[df_join.duplicated(subset=['Id_customer'], keep=False)].index.tolist()
df_join = df_join.drop(index_to_drop)

# Merging customer data (female and male) and credit data
df_merge = pd.merge(df_join,df_credit,on='Id_customer',how='inner')

df_merge.info()


# Replace Overdue Value [1,2,3,4,5] to 1

# In[7]:


df_merge['Overdue'] = df_merge['Overdue'].astype('int64')
df_merge = df_merge.replace({'Overdue': {2: 1, 3: 1, 4: 1, 5: 1}})


# Handling missing values in 'Pekerjaan' column

# In[8]:


# Get data without null value in 'Pekerjaan' column
index_perkerjaan=df_merge[df_merge['Pekerjaan'].isnull()].index.tolist()
df_merge_only = df_merge.drop(index_perkerjaan)

# Get data with only null value in 'Pekerjaan' column
df_pekerjaan = df_merge.iloc[index_perkerjaan]


# In[9]:


# Checking correlation between 'pekerjaan' column to other columns

from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

# Categorical Columns (Pekerjaan)
obj_df = df_merge_only.select_dtypes(include=['object']).columns
for i in obj_df :   
    contingency_table = pd.crosstab(df_merge_only[i], df_merge_only['Pekerjaan'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    #print("Chi-squared {} : {} ".format(i,chi2))
    print("p-value {} : {}".format(i,p))
    print(" ")

# Continuous Columns
obj_df_int = df_merge_only.select_dtypes(include=['int64','float64']).columns
for i in obj_df_int:
    grouped_data = [df_merge_only[i][df_merge_only['Pekerjaan'] == category] for category in df_merge_only['Pekerjaan'].unique()]
    f_statistic, p_value = f_oneway(*grouped_data)
    #print("F-statistic {} : {}".format(i, f_statistic))
    print("p-value {} : {}".format(i,p_value))
    print(" ")


# In[10]:


# GET TRAINING DATA FOR PREDICTING

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_train_work = pd.DataFrame()

obj_df = df_merge_only.loc[:, ['JK','KepemilikanMobil','TipePendapatan','StatusKeluarga','TingkatPendidikan','TipeRumah']]
encoded_columns = pd.DataFrame()
for i in obj_df.columns:   
    encoder = OneHotEncoder(sparse=False, drop='first')  # Drop the first category
    encoded_data = encoder.fit_transform(obj_df[[i]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([i]))
    data_train_work = pd.concat([data_train_work,encoded_df],axis=1)

    #data_train[i] = obj_df[i].astype('category')
    #data_train[i] = data_train[i].cat.codes

list_continuous = ['Pendapatan','Age','JmlAnak','JmlAnggotaKeluarga','Experience']
for i in list_continuous:
    X_continuous = df_merge_only[[i]]
    scaler = MinMaxScaler()
    X_continuous_normalized = scaler.fit_transform(X_continuous)
    # Replace the original continuous column with the normalized values
    data_train_work[i] = df_merge_only[i].reset_index(drop=True)
    #data_train[i] = X_continuous_normalized

# BUILD PREDICTING MODEL

# Convert the 'Category' column to categorical data
df_merge_only['Pekerjaan'] = df_merge_only['Pekerjaan'].astype('category')
pekerjaan_cat = df_merge_only['Pekerjaan'].cat.codes

x = data_train_work
y = pekerjaan_cat

# Resampling data size
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(x, y)

#Split data training and data testing
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# Define Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# Predicting data testing
y_predict = rf_classifier.predict(X_test)

# Get rmse score for evaluation
rmse =mean_squared_error(y_test, y_predict)
print('rmse', rmse)

# Get classification report
class_report = classification_report(y_test, y_predict)
print(class_report)

# Get confusion matrix of predicting result

cm = confusion_matrix(y_test, y_predict)
cm_df = pd.DataFrame(cm,
                     index = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'],
                     columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'])

#Plotting the confusion matrix
plt.figure(figsize=(12,11))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.savefig('Confusion_matrix.png')
plt.show()


# Save 'Pekerjaan' predicted model

# In[11]:


import pickle

# Save the model to a file
pickle_file = "random_forest_model.pkl"
with open(pickle_file, 'wb') as file:
    pickle.dump(rf_classifier, file)
print("\033[1mModel 'Pekerjaan' saved to {}\033[0m".format(pickle_file))


# Predict Missing Value in with saved model

# In[12]:


# LOAD SAVED MODEL
with open(pickle_file, 'rb') as file:
    loaded_model = pickle.load(file)
print("Model loaded from file")

# PREDICTING MISSING VALUES

# Get ready with data input
data_pred = pd.DataFrame()

obj_df = df_pekerjaan.loc[:, ['JK','KepemilikanMobil','TipePendapatan','StatusKeluarga','TingkatPendidikan','TipeRumah']]
encoded_columns = pd.DataFrame()
for i in obj_df.columns:   
    encoder = OneHotEncoder(sparse=False, drop='first')  # Drop the first category
    encoded_data = encoder.fit_transform(obj_df[[i]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([i]))
    data_pred = pd.concat([data_pred,encoded_df],axis=1)

list_continuous = ['Pendapatan','Age','JmlAnak','JmlAnggotaKeluarga','Experience']
for i in list_continuous:
    X_continuous = df_pekerjaan[[i]]
    scaler = MinMaxScaler()
    X_continuous_normalized = scaler.fit_transform(X_continuous)
    # Replace the original continuous column with the normalized values
    data_pred[i] = df_pekerjaan[i].reset_index(drop=True)
    #data_train[i] = X_continuous_normalized
    
# Predict process
predictions = loaded_model.predict(data_pred)

# Fill missing values with predicting values
df_merge['Pekerjaan'] = df_merge['Pekerjaan'].astype('category')
df_pekerjaan['Pekerjaan'] = df_merge['Pekerjaan'].cat.categories[predictions]
df_pekerjaan['Pekerjaan']

# Join df_pekerjaan with df_merge_only
df_merge = pd.concat([df_merge_only, df_pekerjaan])
df_merge.info()


# ## Build Model for Predicting 'Overdue'

# Checking the relation for each column

# In[13]:


# Checking correlation between 'pekerjaan' column to other columns

from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

df_merge['Overdue'] = df_merge['Overdue'].astype('object')
# Categorical Columns (Pekerjaan)
obj_df = df_merge.select_dtypes(include=['object']).columns
for i in obj_df :   
    contingency_table = pd.crosstab(df_merge[i], df_merge['Overdue'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    #print("Chi-squared {} : {} ".format(i,chi2))
    print("p-value {} : {}".format(i,p))
    print(" ")

# Continuous Columns
obj_df_int = df_merge_only.select_dtypes(include=['int64','float64']).columns
for i in obj_df_int:
    grouped_data = [df_merge[i][df_merge['Overdue'] == category] for category in df_merge['Overdue'].unique()]
    f_statistic, p_value = f_oneway(*grouped_data)
    #print("F-statistic {} : {}".format(i, f_statistic))
    print("p-value {} : {}".format(i,p_value))
    print(" ")


# In[17]:


# GET TRAINING DATA FOR PREDICTING

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_train = pd.DataFrame()

obj_df = df_merge.loc[:, ['JK','KepemilikanProperti','TipePendapatan','StatusKeluarga','TipeRumah','Pekerjaan']]
encoded_columns = pd.DataFrame()
for i in obj_df.columns:   
    encoder = OneHotEncoder(sparse=False, drop='first')  # Drop the first category
    encoded_data = encoder.fit_transform(obj_df[[i]])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([i]))
    data_train = pd.concat([data_train,encoded_df],axis=1)

    #data_train[i] = obj_df[i].astype('category')
    #data_train[i] = data_train[i].cat.codes

list_continuous = ['Pendapatan','Age','Experience']
for i in list_continuous:
    X_continuous = df_merge[[i]]
    scaler = MinMaxScaler()
    X_continuous_normalized = scaler.fit_transform(X_continuous)
    # Replace the original continuous column with the normalized values
    data_train[i] = df_merge[i].reset_index(drop=True)
    #data_train[i] = X_continuous_normalized

# BUILD PREDICTING MODEL

# Convert the 'Category' column to categorical data
df_merge['Overdue'] = df_merge['Overdue'].astype('category')
overdue_cat = df_merge['Overdue'].cat.codes

x = data_train
y = overdue_cat

# Resampling data size
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(x, y)

#Split data training and data testing
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size = 0.8)

# Define Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# Predicting data testing
y_predict = rf_classifier.predict(X_test)

# Get rmse score for evaluation
rmse =mean_squared_error(y_test, y_predict)
print('rmse', rmse)

# Get classification report
class_report = classification_report(y_test, y_predict)
print(class_report)

# Get confusion matrix of predicting result

cm = confusion_matrix(y_test, y_predict)
cm_df = pd.DataFrame(cm,
                     index = ['Good Borrower','Bad Borrower'],
                     columns = ['Good Borrower','Bad Borrower'])

#Plotting the confusion matrix
plt.figure(figsize=(9,8))
sns.heatmap(cm_df, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig('Confusion_matrix.png')
plt.show()


# In[18]:


print(len(X_res))
print(len(X_train))
print(len(X_test))


# In[20]:


y_test.value_counts()


# In[ ]:




