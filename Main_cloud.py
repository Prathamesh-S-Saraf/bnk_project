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

# Import Files
from Script.s3_download import s3_download

from Script.preprocessing import pre_processing
from Script.model import fit_score

from Script.s3_upload import s3_upload



s3_down = s3_download()

     
df = pd.read_csv(r'Data\Raw_data\bank-full.csv')



    
def model_pipeline(df):
   
        
    clean_df = pre_processing(df)
    clean_df.to_csv(r'D:\Data science\complete projects\project_bank\Data\clean_df.csv')
    s3_upload()
    print('file uploaded in s3 bucket'.center(50, '*'))
        
    print('Data Cleaning')
    print('Data Cleaning Done '.center(50, '*'))

        
    model = fit_score(clean_df)
    print('model training and evaluation saving successfully')
    
   

if __name__ == "__main__":
    model_pipeline(df)
