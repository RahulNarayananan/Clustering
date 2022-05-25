#importing necessary libraries 
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# read the data
data=pd.read_csv('tshirts.csv')


#extracting values and putting them in an array and joining them
x=np.array(data['height (inches)'].values).reshape(-1,1)
y=np.array(data['weight (pounds)'].values).reshape(-1,1)
z=np.concatenate((x,y),axis=1)


# forming clusters and adding them to the data
km = KMeans(n_clusters=3)
km.fit(z)
y_pred = km.predict(z)
data['cluster']=y_pred


# taking input to predict the tshirt size
X=np.array(float(input('Enter height in inches: ')))
Y=np.array(float(input('Enter weight in pounds: ')))
Z=km.predict([[X,Y]])

if Z==0:
    print('Your ideal size of shirt might be small')
elif Z==1:
    print('Your ideal size of shirt might be medium')
else:
    print('Your ideal size of shirt might be Large')

# providing output in form of graph
data1= data[data.cluster==0]
data2= data[data.cluster==1]
data3= data[data.cluster==2]
plt.scatter(data1['height (inches)'],data1['weight (pounds)'],color='red',label='small')
plt.scatter(data2['height (inches)'],data2['weight (pounds)'],color='blue',label='medium')
plt.scatter(data3['height (inches)'],data3['weight (pounds)'],color='green',label='large')
plt.scatter(X,Y,color='orange')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()


#function to check appropriate number of clusters
def k_checker():
    scale=MinMaxScaler()

    scale.fit(data[['height (inches)']])
    data['height (inches)']=scale.transform(data[['height (inches)']])

    scale.fit(data[['weight (pounds)']])
    data['weight (pounds)']=scale.transform(data[['weight (pounds)']])

    sse = []
    k_rng = range(1,10)
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(data[['height (inches)','weight (pounds)']])
        sse.append(km.inertia_)

    plt.xlabel('K')
    plt.ylabel('Sum of squared error')
    plt.plot(k_rng,sse)
    plt.show()
k_checker()
