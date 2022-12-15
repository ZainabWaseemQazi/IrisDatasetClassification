import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets 
import matplotlib.pyplot

print("Grip: The Spark Foundation")
print("Zainab Waseem Qazi")
print("Data Science and Business Analytics Interee")

#importing data
data_link=r"C:\Users\Haier\Downloads\Iris.csv"
iris_data=pd.read_csv(data_link)
#iris_df=pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
print(iris_data.head(10))
print("Iris Data imported successfully")

#Finding clusters
print("Finding Cluster number for our classification")
num=iris_data.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
var=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(num)
    var.append(kmeans.inertia_)

#iris_df = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
plt.pyplot.plot(range(1, 11), var)
plt.pyplot.title('The Elbow method')
plt.pyplot.xlabel('Number of Clusters')
plt.pyplot.ylabel('Within Cluster sum of square')
plt.pyplot.show()

print("From label we come to know that our elbow occurs near point 3 so we take 3 clusters")
kmeans=KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(num)
# Visualising the clusters - On the first two columns
plt.pyplot.scatter(num[y_kmeans == 0, 0], num[y_kmeans == 0, 1],
            s = 100, c = 'purple', label = 'Iris-setosa')
plt.pyplot.scatter(num[y_kmeans == 1, 0], num[y_kmeans == 1, 1],
            s = 100, c = 'orange', label = 'Iris-versicolour')
plt.pyplot.scatter(num[y_kmeans == 2, 0], num[y_kmeans == 2, 1],
            s = 100, c = 'red', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.pyplot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'black', label = 'Centroids')

plt.pyplot.legend()
plt.pyplot.show()
exit()