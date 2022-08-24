# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv('/content/drive/MyDrive/Cardiovascular Risk Prediction/data_cardiovascular_risk.csv')

#Education null values
df['education'].fillna(df['education'].median(),inplace=True)

#Treating null values
df['BPMeds'] = df['BPMeds'].fillna(df['BPMeds'].median())
df['cigsPerDay'] = df['cigsPerDay'].fillna(df['cigsPerDay'].median())

def fillna_numeric_with_mean(df,col):
  df[col] = df[col].fillna(df[col].mean())

for i in numeric_NA:
  if i!= 'glucose':
    fillna_numeric_with_mean(df,i)  

#KNN to find the missing values for glucose

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Defining scaler and imputer objects
scaler = StandardScaler()
imputer = KNNImputer()

# Imputing missing values with KNN if any
df['glucose'] = imputer.fit_transform((df['glucose'].values.reshape(-1,1)))

df.drop('id',axis=1,inplace=True)

#Z score treatment
def remove_outlier(df,column):
  
  plt.figure(figsize=(15,6))
  plt.subplot(1, 2, 1)
  plt.title('Before Treating outliers')
  sns.boxplot(df[column])
  plt.subplot(1, 2, 2)
  sns.distplot(df[column])
  df = df[((df[column] - df[column].mean()) / df[column].std()).abs() < 3]
  df = df[((df[column] - df[column].mean()) / df[column].std()).abs() > -3]
  
  plt.figure(figsize=(15,6))
  
  plt.subplot(1, 2, 1)
  plt.title('After Treating outliers')
  sns.boxplot(df[column])
  plt.subplot(1, 2, 2)
  sns.distplot(df[column])

df_2 =df.copy()

# herewe can observe that the distribution is not equal
# so we can create another category that
# with 0 cigars = 1st category
# between 1-20 = 2nd category
# between 20-70 = 3rd category
for i in range(len(df)):
  if df['cigsPerDay'][i] == 0:
    df['cigsPerDay'][i] = 'No Cunsumption'
  elif df['cigsPerDay'][i] > 0 and df['cigsPerDay'][i] < 20:
    df['cigsPerDay'][i] = 'Average consumtion'
  else:
    df['cigsPerDay'][i] = 'High Consumption'

df['BP'] = 0

df.loc[(df['sysBP'] < 120) & (df['diaBP'] < 80), 'BP'] = 1

df.loc[((df['sysBP'] >= 120) & (df['sysBP'] < 130)) &
         ((df['diaBP'] < 80)), 'BP'] = 2

df.loc[((df['sysBP'] >= 130) & (df['sysBP'] < 140)) |
         ((df['diaBP'] >= 80) & (df['diaBP'] < 90)), 'BP'] = 3

df.loc[((df['sysBP'] >= 140) & (df['sysBP'] < 180)) |
         ((df['diaBP'] >= 90) & (df['diaBP'] < 120)), 'BP'] = 4

df.loc[(df['sysBP'] >= 180) | (df['diaBP'] >= 120), 'BP'] = 5

cols_BP = ['sysBP', 'diaBP']
df.drop(cols_BP, axis= 1, inplace= True)

#Drop DiaBP as its highly correlated to SysBP, prevalentHyp and diabetes 
#df.drop('diaBP',axis=1,inplace=True)
df.drop('prevalentHyp',axis=1,inplace=True)
df.drop('diabetes',axis=1,inplace=True)


import scipy.stats as stats
import pylab

def to_plot(DF,column):
  plt.figure(figsize=(10,6))
  plt.subplot(1,2,1)
  sns.distplot(DF[column])
  plt.subplot(1,2,2)
  stats.probplot(DF[column],dist='norm',plot=pylab)
  plt.show()

def log_transform(DF,column):
  print("Before Transformation")
  to_plot(DF,column)
  # applying log transformation
  DF[column]=np.log1p(DF[column])
  #plotting
  print("After Transformation")
  to_plot(DF,column)
  # stats.probplot()

def box_cox_transform(DF,column):
  print("Before Transformation")
  to_plot(DF,column)
  # applying boxcox transformation
  DF[column],parameters=stats.boxcox(DF[column])
  print("After Transformation")
  to_plot(DF,column)

numerical_columns = ['age', 'totChol', 'BMI', 'heartRate', 'glucose']
#updating as we deleted sysBP and diaBP
categorical_columns = ['education','cigsPerDay', 'sex', 'is_smoking', 'BPMeds', 'prevalentStroke','TenYearCHD','BP']
#Updating the categorical columns by removing prevaleHyp and diabetes

DF =df[categorical_columns]
#dropping the dependent variable
DF.drop('TenYearCHD',axis=1,inplace=True)
DF = pd.get_dummies(DF, columns=DF.columns)
DF.head()

numerical_columns = ['age', 'totChol', 'BMI', 'heartRate', 'glucose']
#Updated the numerical columns as we deleted 'diaBP'






#Merginf categorical and numerical dependent variables
DF_new = df[numerical_columns].copy()
for i in DF.columns:
  DF_new[i] = DF[i]

#Min max scaler
column_names = list(DF_new.columns)

# column_names

#taking columns to do the minmaxscaling
DF_scaled = pd.DataFrame()


#using standardization as both numeric columns are in different scale

scaler = MinMaxScaler()
scaled = scaler.fit_transform(DF_new[DF_new.columns])
#print(scaled)
DF_scaled = pd.DataFrame(scaler.fit_transform(DF_new[DF_new.columns]))
DF_scaled.columns = column_names  

#We can drop is_smoking column as its highly correlated to consumption of cigarate
DF_scaled.drop('is_smoking_YES',axis=1,inplace=True)
DF_scaled.drop('is_smoking_NO',axis=1,inplace=True)


#Whenever we have only car=tegories then we can drop one as we can its highly negatively correlated
#those columns are sex,BPMeds,prevalenStroke
#So we delete any one category from these
DF_scaled.drop('sex_F',axis=1,inplace=True)
DF_scaled.drop('BPMeds_0.0',axis=1,inplace=True)
DF_scaled.drop('prevalentStroke_0',axis=1,inplace=True)
DF_scaled.drop('cigsPerDay_No Cunsumption',axis=1,inplace=True)

#importing the libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

DF_scaled.columns

X = DF_scaled.copy()
y = df['TenYearCHD'].copy()


#finding the f scores of each features
f_scores = f_regression(X, y)


X = X[selected_features]
y = df['TenYearCHD'].copy()

X_train, X_test, y_train, y_test= train_test_split(X, y,  test_size= 0.30, random_state= 5)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

# the numbers before SMOTE
num_before = dict(Counter(y_train))

#perform SMOTE

# define pipeline
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smote, y_smote = pipeline.fit_resample(X_train, y_train)
#the numbers after SMOTE
num_after =dict(Counter(y_smote))
#using class_wieghts

class_weight = {0: 1,
                1: 6}
##Model Naive bayes                
from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB()


from sklearn.model_selection import RepeatedStratifiedKFold

cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=3, 
                                    random_state=999)



from sklearn.preprocessing import PowerTransformer
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gs_NB = GridSearchCV(estimator=nb, 
                     param_grid=params_NB, 
                     cv=cv_method,
                     verbose=1, 
                     scoring='accuracy')

Data_transformed = PowerTransformer().fit_transform(X_test)

gs_NB.fit(X_test, y_test);

import pickle
filename = 'Cardeo vascular'
pickle.dump(gs_NB,open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict(X_test)