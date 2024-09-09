# import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784')

features, labels = mnist['data'], mnist['target']

rdig = features.loc[28595].values
rdigim = rdig.reshape(28, 28)

plt.imshow(rdigim, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")

#plt.show()
print(labels[28595])

train,test = features[:60000],features[60000:]

trainy,testy = labels[:60000], labels[60000:]

sindex= np.random.permutation(60000)

xtrain,ytrain = train.iloc[sindex],trainy.iloc[sindex]

# ytrain2 = (ytrain==2)
# print(ytrain2)

clf = LogisticRegression(max_iter=1000,tol=0.1)
clf.fit(xtrain,ytrain)

pred = clf.predict(test)
print(pred)

sc = cross_val_score(clf,xtrain,ytrain,cv=3,scoring='accuracy')
print(sc)
print(sc.mean())
