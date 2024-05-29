import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#loading the csv data to a Pandas DataFrame
disease_data = pd.read_csv('C:\\Users\youst\Desktop\machine learning\Train_data.csv')


X= disease_data.drop(columns='Disease', axis=1)
Y= disease_data['Disease']
# Splitting the Data into Training & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)

# Model Training 
# Logistic Regression

model = LogisticRegression()
#training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
#accuracy on training data 
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#accuracy on test data 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('accuracy on Test data :', test_data_accuracy)

# Calculate precision, recall, and F1-score on the test data
precision = precision_score(Y_test, X_test_prediction, average='macro')
recall = recall_score(Y_test, X_test_prediction, average='macro')
f1 = f1_score(Y_test, X_test_prediction, average='macro')

print('Precision on Test data:', precision)
print('Recall on Test data:', recall)
print('F1-score on Test data:', f1)

input_data= (0.7395967125241718,0.6501983879188533,0.7136309861450383,0.8684912414028263,0.6874330284922628,0.5298953992757882,0.2900059089747371,0.6310450180806368,0.0013278578317175,0.7958288704767718,0.0341291220877673,0.0717741989094826,0.1855955968893292,0.0714546096693165,0.6534723763050316,0.5026647785611607,0.2155602381567172,0.5129405631422954,0.0641873469615352,0.610826509528389,0.9394848536044538,0.0955115282493801,0.4659569674775698,0.7692300746279673)
#change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)
#reshape the numpy array as we are predicting for only instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
