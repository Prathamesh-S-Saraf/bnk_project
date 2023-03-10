#!/usr/bin/env python
# coding: utf-8

# ## 1. Libraries

# In[46]:


import pandas as pd
import numpy as np
import re

#visualisation Libraries
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px
#%matplotlib inline

#Libraries to build model
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.tree   import DecisionTreeClassifier
from sklearn.ensemble  import AdaBoostClassifier
from sklearn.metrics  import confusion_matrix ,accuracy_score, precision_score,recall_score
from sklearn.metrics import classification_report, roc_curve, f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from skfeature.function.similarity_based import fisher_score
#from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from datetime import date
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek


# ## 2. Problem Statement

# Bank wants to improve its term deposit subscription rate by accurately predicting which of its clients will subscribe to a 
# term deposit. The bank has a large client database that includes various demographic, financial, and transactional
# information about its clients. However, despite this information, the bank still struggles to predict which clients 
# will subscribe to a term deposit, leading to missed opportunities and reduced profits. The goal of this project is to 
# develop a machine learning model that can accurately predict which clients are likely to subscribe to a term deposit. 
# This will enable the bank to target its marketing efforts more effectively, resulting in increased term deposit 
# subscriptions and higher profits.

# ## 3.Data Gathering

# In[47]:


bnk_mark_df = pd.read_csv(r"D:\Data science\project\ML\New folder\bank-full.csv")
bnk_mark_df


# ## 4. Pre-processing

# In[74]:


def pre_processing(bnk_mark_df): 
    
    #AGE 
    bnk_mark_df['age'] = np.where(bnk_mark_df['age']>=70 ,bnk_mark_df['age'].median(),bnk_mark_df['age'])
    encoder=LabelEncoder()
    
    #JOB
    bnk_mark_df['job']=encoder.fit_transform(bnk_mark_df['job'])
    bnk_mark_df['job']
    
    #MARITAL
    bnk_mark_df['marital'].replace({'married': 0, 'single': 1, 'divorced': 2}, inplace=True)
    
    #EDUCATION
    bnk_mark_df['education'].replace({'secondary': 2, 'tertiary': 3, 'primary':1, 'unknown':0}, inplace=True)
    bnk_mark_df['education'] = np.where(bnk_mark_df['education']<=0.5 ,bnk_mark_df['education'].median(),bnk_mark_df['education'])
    
    #DEFAULT
    bnk_mark_df['default'].replace({'no': 0, 'yes': 1}, inplace=True)
    
    #BALANCE
    bnk_mark_df['balance'] = np.where(bnk_mark_df['balance']>=45000 ,bnk_mark_df['balance'].median(),bnk_mark_df['balance'])
    bnk_mark_df['balance'] = np.where(bnk_mark_df['balance']<=-2000 ,bnk_mark_df['balance'].median(),bnk_mark_df['balance'])
    
    #HOUSNING
    bnk_mark_df['housing'].replace({'no': 0, 'yes': 1}, inplace=True)
    
    #LOAN
    bnk_mark_df['loan'].replace({'no': 0, 'yes': 1}, inplace=True)
    
    #CONTACT
    bnk_mark_df['contact'].replace({'cellular': 1, 'unknown': 2, 'telephone': 3}, inplace=True)
    
    #MONTH
    bnk_mark_df['month'].replace({'may': 5,
     'jul': 7,
     'aug': 8,
     'jun': 6,
     'nov': 11,
     'apr': 4,
     'feb': 2,
     'jan': 1,
     'oct': 10,
     'sep': 9,
     'mar': 3,
     'dec': 12}, inplace=True)
    
    #DURATION
    bnk_mark_df['duration'] = np.where(bnk_mark_df['duration']>=3000 ,bnk_mark_df['duration'].median(),bnk_mark_df['duration'])
    
    #CAMPAIGN
    bnk_mark_df['campaign'] = np.where(bnk_mark_df['campaign']>=45 ,bnk_mark_df['campaign'].median(),bnk_mark_df['campaign'])
    
    #PREVIOUS
    bnk_mark_df['previous'] = np.where(bnk_mark_df['previous']>=45 ,bnk_mark_df['previous'].median(),bnk_mark_df['previous'])
    
    #PREVIOUS_OUTCOME
    bnk_mark_df['previous_outcome'].replace({'unknown': 3, 'failure': 0, 'other': 2, 'success': 1}, inplace=True)
    
    #OFFER_STATUS
    bnk_mark_df['offer_status'].replace({'no': 0, 'yes': 1}, inplace=True)
     
    #CSV
    
    pre_process_df.to_csv(r'C:\Users\Prathamesh\Desktop\project_bank\Data\clean_df.csv')
    return bnk_mark_df
pre_process_df = pre_processing(bnk_mark_df)
pre_process_df


# ## 5. Model Training and eveluation.

# In[72]:


def fit_score(pre_process_df):
    global x
    global y
    global min_train_x
    global min_test_y
    global x_train
    global y_train
    # TRAIN_TEST_SPLIT
    x = bnk_mark_df.drop(['age','balance','contact','day','month','duration','passed_days','offer_status'], axis=1)
    y = bnk_mark_df[['offer_status']]
    x_train,x_test,y_train,y_test =train_test_split(x,y, test_size=0.2,random_state=35,stratify=y)
    
    # CSV
    x_train.to_csv(r'C:\Users\Prathamesh\Desktop\project_bank\Data\Processed_df\Train\x_train.csv')
    y_train.to_csv(r'C:\Users\Prathamesh\Desktop\project_bank\Data\Processed_df\Train\y_train.csv')
    x_test.to_csv(r'C:\Users\Prathamesh\Desktop\project_bank\Data\Processed_df\Test\x_test.csv')
    y_test.to_csv(r'C:\Users\Prathamesh\Desktop\project_bank\Data\Processed_df\Test\y_test.csv')
    
    # FEATURE_SCALING (MIN_MAX_SCALER)
    min_scaler = MinMaxScaler()
    #min_scaler.fit(x_train)                                     
    min_train_x = min_scaler.fit_transform(x_train)
    min_test_y = min_scaler.transform(x_test)
    
    # SMOTE
    smote_os = SMOTE(sampling_strategy='auto')
    x_train_sm, y_train_sm = smote_os.fit_resample(min_train_x,y_train)
    print(y_train.value_counts())
    print(y_train_sm.value_counts())

    global rf_clf
    #MODEL_BUILDING (RANDOM_FOREST)
    rf_clf = RandomForestClassifier(random_state=11, n_estimators = 30, oob_score=True, n_jobs=-1)
    rf_clf=rf_clf.fit(x_train,y_train)
    print('Model Executed '.center(50, '*'))
    print('RANDOM FOREST')
    
    

   # Model Evaluation 
    def model_evaluation(algo, ind_var, y_act ):
        model=algo
        pred = model.predict(ind_var)
    
        accuracy_rate = accuracy_score(y_act, pred)
        print('Accuracy of model is : ',accuracy_rate)
        print()

        conf_matrix = confusion_matrix(y_act, pred)
        print('confusion matrix is : \n', conf_matrix)
        print()

        clsf_report = classification_report(y_act, pred)
        print('classification report is : \n', clsf_report)
    
        return pred,model_evaluation

    print('Test evaluation data '.center(50, '*'))
    print()
    model_evaluation(rf_clf, x_test, y_test)
    print()
    print()
    print('Train evaluation data '.center(50, '*'))
    print()
    model_evaluation(rf_clf, x_train, y_train)


random_forest_clf = fit_score(pre_process_df)
random_forest_clf


# ## 6.Testing on single row

# In[50]:


x.head(1).T


# In[51]:


job = 'management'
marital = 'married'
education = 'primary'
default  = 'no'
housing  = 'yes'
loan    = 'no'
campaign = 4.0
previous  = 2.0
previous_outcome = 'failure'


# offer_subscription = ?



# In[52]:


test_array = np.array([4.0, 1.0, 1.0, 0.0,1.0, 0.0, campaign ,previous,0.0 ], ndmin = 2)
test_array


# In[53]:


rf_clf.predict(test_array)


# In[54]:


test_array = np.array([job,marital,education,default,housing,loan,campaign,previous,previous_outcome],ndmin=2)
test_array


# In[55]:


print(x.columns)
len(x.columns)


# In[56]:


x.shape[1]


# In[57]:


job_dict = {'admin':0,'blue-collar':1,'enterpreneur':2,'housemade':3,'management':4,'retired':5,
           'self-employed':6,'services':7,'student':8,'technician':9,'unemployed':10,'unknown':11}


# In[58]:


marital_dict = {'married':0,'single':1,'divorced':2}


# In[59]:


education_dict = {'unknown':0, 'primary':1,'secondary':2,'tertiary':3}


# In[60]:


default_dict = {'yes':1,'no':0}


# In[61]:


housing_dict = {'yes':1,'no':0}


# In[62]:


loan_dict = {'yes':1,'no':0}


# In[63]:


previous_outcome_dict = {'unknown':3,'failure':0,'success':1,'other':2}


# In[64]:


project_data = {'job': job_dict ,
               'marital': marital_dict ,
                'education' : education_dict,
                'default' : default_dict,
                'housing' : housing_dict,
                'loan'     :loan_dict,
                'previous_outcome' : previous_outcome_dict,
                'columns': list(x.columns)}
project_data


# In[65]:


test_array = np.zeros(x.shape[1])
test_array[0] = project_data['job'][job]
test_array[1] = project_data['marital'][marital]
test_array[2] = project_data['education'][education]
test_array[3] = project_data['default'][default]
test_array[4] = project_data['housing'][housing]
test_array[5] = project_data['loan'][loan]
test_array[6] = campaign
test_array[7] = previous
test_array[8] =  project_data['previous_outcome'][previous_outcome]

test_array


# In[66]:


result = np.around(rf_clf.predict([test_array]), 2)
print(f'offer subscription status is :{result}')


# In[67]:


test_array


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 6. Model upload

# In[68]:


import joblib
import pickle
with open('random_forest_clf.pkl', 'wb') as file:
    pickle.dump(rf_clf, file)

with open('random_forest_clf.pkl', 'rb') as file:
    pickle.load(file)


# In[69]:


import json
with open('project_data.json', 'w') as f:
    json.dump(project_data, f)


# In[ ]:




