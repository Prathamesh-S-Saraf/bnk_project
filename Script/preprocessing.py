## 1. Libraries
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

## 2. Problem Statement
#Bank wants to improve its term deposit subscription rate by accurately predicting which of its clients will subscribe to a 
#term deposit. The bank has a large client database that includes various demographic, financial, and transactional
#information about its clients. However, despite this information, the bank still struggles to predict which clients 
#will subscribe to a term deposit, leading to missed opportunities and reduced profits. The goal of this project is to 
#develop a machine learning model that can accurately predict which clients are likely to subscribe to a term deposit. 
#This will enable the bank to target its marketing efforts more effectively, resulting in increased term deposit 
#subscriptions and higher profits.


## 3.Data Gathering
bnk_mark_df = pd.read_csv(r"D:\Data science\complete projects\project_bank\Data\Raw_data\bank-full.csv")
bnk_mark_df

## 4. Pre-processing

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
     
    
    return bnk_mark_df
# pre_processing(bnk_mark_df)



