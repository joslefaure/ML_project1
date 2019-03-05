import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from urllib.request import urlretrieve
import numpy as np
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image  





iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

urlretrieve(iris)
df = pd.read_csv(iris, header = None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 

# print (df)

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "Species"]
att = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df.columns = attributes
# print(df)
# print(df.shape)
arr = df.values
###############
x = []
for row in range (0, len(arr)):
	x.append(arr[row][0])
	sl_sum = sum(x)
###############
y = []
for row in range (0, len(arr)):
	y.append(arr[row][1])
	sw_sum = sum(y)
################
z = []
for row in range (0, len(arr)):
	z.append(arr[row][2])
	pl_sum = sum(z)
################
w = []
for row in range (0, len(arr)):
	w.append(arr[row][3])
	pw_sum = sum(w)
###############
values_count = np.array([sl_sum, sw_sum, pl_sum, pw_sum])
x_pos = [1, 2, 3, 4]
average = df.mean()
sd = df.std()

# print the calculated values
print ("The AVG of the features")
print(average)
print("The SD of the features")
print(sd)
print("Counts of the features respectively")
print(sl_sum)
print(sw_sum)
print(pl_sum)
print(pw_sum)



avg_list = average.values
sd_list = sd.values


# show the mean and SD
fig, ax = plt.subplots()
ax.bar(x_pos, avg_list, yerr=sd_list, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Y axis')
ax.set_xticks(x_pos)
ax.set_xticklabels(att)
ax.set_title('Mean and Standard deviation of the 4 features')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

# show the value counts
fig1, ax1 = plt.subplots()
ax1.bar(x_pos, values_count, align='center', alpha=0.5, ecolor='black', capsize=10)
ax1.set_ylabel('Counts')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(att)
ax1.set_title('Plot of the values count of the features')
ax1.yaxis.grid(True)
plt.tight_layout()
plt.show()

# Part 2

# importing all the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.neighbors import KNeighborsClassifier # KNN classifier
from sklearn import svm # for suport vector machine algorithm
from sklearn import metrics # for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier # for using DTA
from sklearn import tree
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
# print(df)
data = np.array(df)
X = data[:, :-1]
y = data[:,-1]

# train, test = train_test_split(df, test_size=0.3) # our main data split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%


# train_X = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
# train_y = train.Species # output of the training data

# test_X = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data feature
# test_y = test.Species # output value of the test data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.30)

# Decision tree
model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
accuracy = metrics.accuracy_score(prediction, test_y)
print("Accuracy of DT for Iris : ", accuracy)

#CM
print ("The Confusion matrix: ")
print (confusion_matrix(test_y, prediction))

# resubstitution validation
prediction_identical = model.predict(train_X)
print('The accuracy resubstitution: ', metrics.accuracy_score(prediction_identical, train_y))
# Kfold
kf = KFold(n_splits=2)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
   print("KFold train-test")
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
## Check Correlation
plt.figure(figsize=(8,4))
sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r') # draws heatmap with input as correlation matrix calculated by iris.corr() 
plt.show()

# 
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=att)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())
# Create PDF
graph.write_pdf("iris.pdf")

# Create PNG
graph.write_png("iris.png")