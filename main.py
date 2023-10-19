import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump, load

df = pd.read_csv('url_data.csv')

lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])
df["type_code"].value_counts()

X = df['url'].values
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
dump(vectorizer, open('vectorizer.pkl', "wb"))

y = df['type_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=8)

model = SGDClassifier(loss='hinge', penalty='l2',
                      fit_intercept=True, shuffle=True)

# Fitting the data in the model

# Using fit will overwrite the previous weights on each epoch
# Partial fit enables fitting new data to the previously trained model

model.partial_fit(X_train, Y_train, classes=np.unique(Y))

# Training the model with training data and printing it's accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# Testing the model with testing data and printing it's accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

# Saving the model (Pickling) so that it can be used to predict data or trained with new data

dump(model, open("model.pkl", "wb"))