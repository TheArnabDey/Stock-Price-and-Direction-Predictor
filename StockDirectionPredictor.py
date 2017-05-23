#import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
#load the dataset into the dataframe
df = pd.read_csv('PTG_Data_V1.1.csv')
#drop COL076 as it's NULL
df = df.drop("COL076", axis=1)
#define a new dataframe taking only date and time time from the original dataset
df1 = df.iloc[:,0:2]
#Define individual Month, Date, Year and Hour columns which we are going to extract from the imported Date and Time columns
df1["Month"] = 0
df1["Date"] = 0
df1["Year"] = 0
df1["Hour"] = 0
#Extraction of date and time and copy individual components to respective columns
for i in range(0,len(df1)):
    a = df1.iloc[i,0].split('/')
    df1.iloc[i,2] = a[0]
    df1.iloc[i, 3] = a[1]
    df1.iloc[i, 4] = a[2]
    b = df1.iloc[i, 1].split(':')
    df1.iloc[i, 5] = b[0]
#converting all columns to int as these columns came from string splitting
df1.iloc[:, 2] = df1.iloc[:, 2].astype(int)
df1.iloc[:,3] = df1.iloc[:,3].astype(int)
df1.iloc[:,4] = df1.iloc[:,4].astype(int)
df1.iloc[:,5] = df1.iloc[:,5].astype(int)

#do the same things for the test/prediction data
#load the dataset into the dataframe
dft = pd.read_csv('PTG_Data_V3.1.csv')
#define a new dataframe taking only date and time time from the original dataset
df1t = dft.iloc[:,0:2]
#Define individual Month, Date, Year and Hour columns which we are going to extract from the imported Date and Time columns
df1t["Month"] = 0
df1t["Date"] = 0
df1t["Year"] = 0
df1t["Hour"] = 0
#Extraction of date and time and copy individual components to respective columns
for i in range(0,len(df1t)):
    a = df1t.iloc[i,0].split('/')
    df1t.iloc[i,2] = a[0]
    df1t.iloc[i, 3] = a[1]
    df1t.iloc[i, 4] = a[2]
    b = df1t.iloc[i, 1].split(':')
    df1t.iloc[i, 5] = b[0]
#converting all columns to int as these columns came from string splitting
df1t.iloc[:, 2] = df1t.iloc[:, 2].astype(int)
df1t.iloc[:,3] = df1t.iloc[:,3].astype(int)
df1t.iloc[:,4] = df1t.iloc[:,4].astype(int)
df1t.iloc[:,5] = df1t.iloc[:,5].astype(int)
#define training feature columns
train_features = df1.iloc[:,2:6]
#define training labels column
train_labels = df.iloc[:,77]
#define test/prediction feature columns
test_features = df1t.iloc[:,2:6]
#define the classifier
clf = GradientBoostingClassifier()
#fit the classifier to training data
clf.fit(train_features,train_labels)
#predict the values of the test/prediction set
pred = clf.predict(test_features)
#Create the output dataframe by keeping only date, time and prediction values
df1t["TargetClassification"] = pred
df1t = df1t.drop("Month", axis=1)
df1t = df1t.drop("Date", axis=1)
df1t = df1t.drop("Year", axis=1)
df1t = df1t.drop("Hour", axis=1)
#Writing the output dataframe to a file
df1t.to_csv('TargetClassificationPrediction.csv', index= False)
