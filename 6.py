import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('6_dataset.csv')

print("The first 5 values of data is :\n", data.head())

X = data.iloc[:, :-1]
print("\nThe First 5 values of train data is\n", X.head())

y = data.iloc[:, -1]
print("\nThe first 5 values of Train output is\n", y.head())

le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)

le_temperature = LabelEncoder()
X.Temperature = le_temperature.fit_transform(X.Temperature)

le_humidity = LabelEncoder()
X.Humidity = le_humidity.fit_transform(X.Humidity)

le_windy = LabelEncoder()
X.Windy = le_windy.fit_transform(X.Windy)

print("\nNow the Train data is :\n", X.head())

le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)

print("\nNow the Train output is\n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

print("Accuracy is:", accuracy_score(classifier.predict(X_test), y_test))




# OUTPUT The first 5 values of data is :
#  Outlook Temperature Humidity Windy PlayTennis
# 0 Sunny Hot High Weak No
# 1 Sunny Hot High Strong No
# 2 Overcast Hot High Weak Yes
# 3 Rain Mild High Weak Yes
# 4 Rain Cool Normal Weak Yes
# The First 5 values of train data is
#  Outlook Temperature Humidity Windy
# 0 Sunny Hot High Weak
# 1 Sunny Hot High Strong
# 2 Overcast Hot High Weak
# 3 Rain Mild High Weak
# 4 Rain Cool Normal Weak

# The first 5 values of Train output is
# 0 No
# 1 No
# 2 Yes
# 3 Yes
# 4 Yes
# Name: PlayTennis, dtype: object
# Now the Train data is :
#  Outlook Temperature Humidity Windy
# 0 2 1 0 1
# 1 2 1 0 0
# 2 0 1 0 1
# 3 1 2 0 1
# 4 1 0 1 1
# Now the Train output is
# [0 0 1 1 1 0 1 0 0 1 1 1 1 0]
# Accuracy is: 0.6666666666666666