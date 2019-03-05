import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from urllib.request import urlretrieve
import numpy as np
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.neighbors import KNeighborsClassifier # KNN classifier
from sklearn import metrics # for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier # for using DTA
from sklearn import tree
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


df = pd.read_csv('googleplaystore.csv')

# let's do some cleaning
df = df.rename(index=str, columns={"Content Rating": "Content_rating", "Last Updated": "Last_updated", "Current Ver":"Current_ver", "Android Ver":"Android_ver"})
df['Price'] = df.Price.str.replace('Everyone','0')
df['Price'] = df.Price.str.replace('$','')
df['Android_ver'] = df.Android_ver.str.replace('[^\d.]','')
df['Installs'] = df.Installs.str.replace('[^\d.]','')
df.mask((df == "Varies with device") | (df == "Unrated"), inplace=True)

# removing a parasite row
df = df.drop(df.index[[10472]])

# convert the continuous features to float
df.Rating = df.Rating.astype(float)
df['Reviews'] = df['Reviews'].str.extract('(\d*\.?\d*)', expand=False).astype(float)
df['Installs'] = pd.to_numeric(df.Installs.str.replace('[^\d.]', ''), errors='coerce')
df['Price'] = df['Price'].str.extract('(\d*\.?\d*)', expand=False).astype(float)

# the feature size required more tedious work
def preprocess_Size(df_size):
	df_size = (df_size.replace(r'[kM]+$', '', regex=True).astype(float) * \
				df_size.str.extract(r'[\d\.]+([kM]+)', expand=False)
				.fillna(1)
				.replace(['k','M'], [10**3, 10**6]).astype(int))
	return(df_size)
df['Size'] = preprocess_Size(df['Size'])


# Calculate the means of the continuous features
avg_rating = df["Rating"].mean()
avg_size = df["Size"].mean()
avg_reviews = df["Reviews"].mean()
avg_install = df["Installs"].mean()
avg_price = df["Price"].mean()
# Put the averages in a list
avg_list = [avg_rating, avg_reviews, avg_size, avg_install, avg_price]

# Print the means
print("~ The features Avg ~")
print("avg_rating: ", avg_rating)
print("avg_reviews: ", avg_reviews)
print("avg_install: ", avg_install)
print("avg_price: ", avg_price)
print("avg_size: ", avg_size)
# print("Rating avg: ", avg_rating)

# Calculate the SD
sd_rating = df["Rating"].std()
sd_review = df["Reviews"].std()
sd_size = df["Size"].std()
sd_install = df["Installs"].std()
sd_price = df["Price"].std()
# Put them in a list
std_list = [sd_rating, sd_review, sd_size, sd_install, sd_price]
# print the SDs
print("~ The SD of the features ~")
print("Rating: ", sd_rating)
print("Reviews: ", sd_review)
print("Size: ", sd_size)
print("Installs: ", sd_install)
print("Price: ", sd_price)


x_pos = [1, 2, 3, 4, 5]
att = ["Rating", "Reviews", "Size", "Installs", "Price"]


# show the mean and SD
fig, ax = plt.subplots()
ax.bar(x_pos, avg_list, yerr=std_list, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Y axis')
ax.set_xticks(x_pos)
ax.set_xticklabels(att)
ax.set_title('Mean and Standard deviation of the continuous features')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

# Value count
values_count = [df["Rating"].sum(), df["Reviews"].sum(), df["Size"].sum(), df["Installs"].sum(), df["Price"].sum()]
fig1, ax1 = plt.subplots()
ax1.bar(x_pos, values_count, align='center', alpha=0.5, ecolor='black', capsize=10)
ax1.set_ylabel('Counts')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(att)
ax1.set_title('Plot of the values count of the continuous features')
ax1.yaxis.grid(True)
plt.tight_layout()
plt.show()

# Preprocess all the data (from categorical to continuous)
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype!= np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1
			df[column]= list(map(convert_to_int,df[column]))

	return df

df = handle_non_numerical_data(df)

# Check correlation
# plt.figure(figsize=(8,4))
# sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r') # draws heatmap with input as correlation matrix calculated by iris.corr() 
# plt.show()

# Delete useless column
del df["App"]
# Since we will use Category as target, put it as the last column
df = df[['Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content_rating', 'Genres', 'Last_updated', 'Current_ver', 'Android_ver', 'Category']]
features = ['Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content_rating', 'Genres', 'Last_updated', 'Current_ver', 'Android_ver']
# Reconvert to csv for Random Forest algo
df_preprocessed = df.to_csv('googleplaystore_preprocessed.csv', index=False)


# Preprocessing
data = np.array(df)
X = np.delete(data, 11, axis=1)
X = np.nan_to_num(X)
y = data[:,-1]

# Splitting
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# Decision tree
model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
accuracy = metrics.accuracy_score(prediction, test_y)
print("Accuracy of DT : ", accuracy)

# plot DT
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=features)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())
# Create PDF and PNG
graph.write_pdf("googleplaystore_tree.pdf")
graph.write_png("googleplaystore_tree.png")


# Confusion Matrix
print ("The Confusion matrix: ")
print (confusion_matrix(test_y, prediction))
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(print("TP, FP, TN, FN: ",TP, FP, TN, FN))
print(perf_measure(test_y, prediction))
# Kfold
kf = KFold(n_splits=4)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
   print("KFold train-test cross validation")
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
# resubstitution validation
prediction_identical = model.predict(train_X)
print('The accuracy resubstitution: ', metrics.accuracy_score(prediction_identical, train_y))






