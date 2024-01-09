import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

PlayTennis = pd.read_csv("4_dataset.csv")

Le = LabelEncoder()
PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy'] = Le.fit_transform(PlayTennis['windy'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])

y = PlayTennis['play']
X = PlayTennis.drop(['play'], axis=1)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

tree.plot_tree(clf)

X_pred = clf.predict(X)
print(X_pred == y)


# Datasets: Filename: PlayTennis.csv
# outlook,temp,humidity,windy,play
# Sunny,Hot,High,Weak,No
# Sunny,Hot,High,Strong,No
# Overcast,Hot,High,Weak,Yes
# Rain,Mild,High,Weak,Yes
# Rain,Cool,Normal,Weak,Yes
# Rain,Cool,Normal,Strong,No
# Overcast,Cool,Normal,Strong,Yes
# Sunny,Mild,High,Weak,No
# Sunny,Cool,Normal,Weak,Yes
# Rain,Mild,Normal,Weak,Yes
# Sunny,Mild,Normal,Strong,Yes
# Overcast,Mild,High,Strong,Yes
# Overcast,Hot,Normal,Weak,Yes
# Rain,Mild,High,Strong,No

# Output:
# 0 True
# 1 True
# 2 True
# 3 True
# 4 True
# 5 True
# 6 True
# 7 True
# 8 True
# 9 True
# 10 True
# 11 True
# 12 True
# 13 True
