import pandas as pd
dataset=pd.read_csv('G:\\data\\20news-18828.csv',header=None,delimiter=',',names=['label','text'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset.text,dataset.label,test_size=0.3)
print(x_train.shape)
print(x_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

cv=CountVectorizer()
x_train_counts=cv.fit_transform(x_train)
x_test_counts=cv.transform(x_test)
estimator=MultinomialNB()
estimator.fit(x_train_counts,y_train)
predicted=estimator.predict(x_test_counts)

#print(classification_report(y_test,predicted,labels=dataset.label.unique()))

from sklearn.metrics import classification_report,precision_recall_fscore_support
import re
def normalize_numbers(s):
    return re.sub(r'\b\d+\b','NUM',s)
cv=CountVectorizer(preprocessor=normalize_numbers,stop_words='english')
x_train_counts=cv.fit_transform(x_train)
x_test_counts=cv.transform(x_test)
estimator=MultinomialNB()
estimator.fit(x_train_counts,y_train)
predicted=estimator.predict(x_test_counts)
print('CountVectorizer处理特征结果')
print(classification_report(y_test,predicted,labels=dataset.label.unique()))
from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(preprocessor=normalize_numbers,stop_words='english')
x_train_tf=tv.fit_transform(x_train)
x_test_tf=tv.transform(x_test)
estimator2=MultinomialNB()
estimator2.fit(x_train_tf,y_train)
predicted2=estimator2.predict(x_test_tf)
print('TfidfVectorizer处理特征结果')
print(classification_report(y_test,predicted2,labels=dataset['label'].unique()))
from sklearn.linear_model import LogisticRegression
ev=CountVectorizer()
x_train_ev=ev.fit_transform(x_train)
x_test_ev=ev.transform(x_test)
LR=LogisticRegression()
LR.fit(x_train_ev,y_train)
predicted3=LR.predict(x_test_ev)
print('LogisticRegression模型分类结果')
print(classification_report(y_test,predicted3,labels=dataset['label'].unique()))

from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import preprocessing
x_train,x_test,y_train,y_test=train_test_split(dataset.text,dataset['label'],test_size=0.3,random_state=100)
cv=CountVectorizer(max_features=200)
x_train_counts=cv.fit_transform(x_train)
x_test_counts=cv.transform(x_test)
kmeans=KMeans(n_clusters=20,random_state=100)
kmeans.fit(x_test_counts)
print(kmeans.labels_)
print(kmeans.predict(x_test_counts))

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
x=dataset.text
y=dataset.label
kf=KFold(n_splits=5,shuffle=True)
for train_index,test_index in kf.split(x):
    x_train,x_test,y_train,y_test=x[train_index],x[test_index],y[train_index],y[test_index]
    cv=CountVectorizer()
    x_train_counts=cv.fit_transform(x_train)
    x_test_counts=cv.transform(x_test)
    clf=MultinomialNB()
    clf.fit(x_train_counts,y_train)
    predicted=clf.predict(x_test_counts)
    p,r,f1,_=precision_recall_fscore_support(y_test,predicted,average='macro')
    print("\n准确率:{0},召回率:{1},F1值:{2}".format(p,r,f1))





