#Import Libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
#Load the Dataset
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
#data= pd.read_csv("D:/GEO/BE COURSES/LAB/DATASET/pima-indians-diabetes.csv")
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
#Splitting Data into Training and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)
#Scikit-learn 4-Step Modeling Pattern
# Step 1: Import the model you want to use
# This was already imported earlier in the notebook so commenting out
#from sklearn.tree import DecisionTreeClassifier
# Step 2: Make an instance of the Model
clf = DecisionTreeClassifier(max_depth = 2,random_state = 0)
# Step 3: Train the model on the data
clf.fit(X_train, Y_train)
# Step 4: Predict labels of unseen (test) data
# Not doing this step in the tutorial
# clf.predict(X_test)
#How to Visualize Decision Trees using Matplotlib
tree.plot_tree(clf);

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')
