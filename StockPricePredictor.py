#import all the required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
#load the training dataset into the dataframe
df = pd.read_csv('PTG_Data_V1.1.csv')
#drop COL076 as it's NULL
df = df.drop("COL076", axis=1)
#define training feature columns
train_features = df.iloc[:,9:11]
#define training labels columns
train_labels = df.iloc[:,76]
#load the prediction dataset into the dataframe
dft = pd.read_csv('PTG_Data_V3.1.csv')
#define prediction feature columns
test_features = dft.iloc[:,9:11]
#define the classifier
clf = LinearRegression()
#fit the classifier to training data
clf.fit(train_features,train_labels)
#predict the values for the test/prediction set
pred = clf.predict(test_features)
#Creating an output dataframe and storing date, time and prediction values to it
dfp = dft.iloc[:,0:2]
dfp["TargetRegression"] = pred
#Writing the output dataframe to a csv file
dfp.to_csv('TargetRegressionPrediction.csv', index= False)
