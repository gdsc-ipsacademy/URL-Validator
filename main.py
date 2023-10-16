import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('url_data.csv')
X = df['url'].values
y = df['type'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=8)