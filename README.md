# Bobble.Ai
AI/interview assignment solution

Part A: MCQ : -

1 . (B) 19/59
2 . (B) False Positive
3 . (B)
4 . (D) 
5 . (C)
6 . (D)
7 . (A)
8 . (D)
9 . (D)
10. (C)
11. (B)
12. (D)
13. (B)
14. (B)
15. (C)
16. (B)
17. (A)
18. (C)
19. (D)
20. (C)
21. (B)
22. (B)
23. (B)
24. (B)
25. (A)

Part B:Fill in the Blanks

1 . Decrease
2 . Non Linear , Regression , Decrease
3 . Classification
4.  Testing , Training
5. -1 to 256

Part C:Long Questions

1.) solution using Python
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
nltk.download('punkt')


a="Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when hewas 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers.He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University.Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America.He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college. "
#1
b=a.lower()
print(b)

c = ""
p = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
for i in a:
   if i not in p:
       c=c+i
print(c)

def remove(a): 
    return a.replace(" ", "") 
print(remove(a)) 
sw = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(a) 
  
filtered_sentence = [w for w in word_tokens if not w in sw] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in sw:
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence)
ps=PorterStemmer()
f= a.split()
for j in f: 
    print(w, " : ", ps.stem(w))


3.)

def check(matrix):
        iscol = False
        row = len(matrix)
        col = len(matrix[0])
        for i in range(row):
            if matrix[i][0] == 0:
                iscol = True
            for j in range(1, col):
                if matrix[i][j]  == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0

        for i in range(1, row):
            for j in range(1, Col):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j] = 0


        if matrix[0][0] == 0:
            for j in range(col):
                matrix[0][j] = 0
        if iscol:
            for i in range(row):
                matrix[i][0] = 0
        print(matrix)
matrix = [[0,1,1,0],
[1,0,1,1],
[0,0,1,1],
[1,0,1,0]]
print(check(matrix))

5.)


from google.colab import drive
drive.mount('/content/drive/')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random

t_data = pd.read_csv('/content/drive/My Drive/bobble/train.csv')
v_data = pd.read_csv('/content/drive/My Drive/bobble/validation.csv')
t_data['target']=[1 if x=='silent' or x=='noise'  else 0 for x in t_data['target']]
v_data['target']=[1 if x=='silent' or x=='noise'  else 0 for x in v_data['target']]
i_train=t_data.drop('target',1)
d_train=t_data.target
i_valid=v_data.drop('target',1)
d_valid=v_data.target
del i_train['filename']
del i_valid['filename']

import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import warnings; warnings.simplefilter('ignore') 
models = []
models.append(('Logistic Regression', LogisticRegression()))

models.append(('Decison Tree', DecisionTreeClassifier()))

models.append(('SVM', SVC()))

for name,model in models:
    
    
    classifier = model
    classifier.fit(i_train, d_train)

    pred = classifier.predict(i_valid)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(d_valid, pred)

   
  
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = i_train, y = d_train, cv = 10)
    accuracies.mean()
    accuracies.std()
    
    print()
    print("For {0} The Performance result is: ".format(name))
    print()

    #the performance of the classification model
    print("the Accuracy is: "+ str((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])))
    recall = cm[1,1]/(cm[0,1]+cm[1,1])
    print("Recall is : "+ str(recall))
    print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
    precision = cm[1,1]/(cm[1,0]+cm[1,1])
    print("Precision is: "+ str(precision))
    print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))
    from math import log
    print("Entropy is: "+ str(-precision*log(precision)))
