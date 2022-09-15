from cProfile import label
import csv
from matplotlib import projections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from mpl_toolkits import mplot3d


read_path = r"Cor_av_CvN_20221800"
time = "1800" ## change this to access the other time windows 

write_path = r"C:\Users\adsto\OneDrive\Documents\UCL\Individual project\Final Report\Figures\02_07_2022\1800s_timeSeries_ml.csv"

graph_save_path = r"C:\Users\adsto\OneDrive\Documents\UCL\Individual project\Final Report\Figures\02_07_2022\n"
graph = graph_save_path+time

all_data = pd.read_csv(read_path, index_col="index") #,index_col="Unnamed: 0"
# #print(all_data)
# df = pd.read_csv("data_frame2022_",index_col="index")
# #print(all_data)

all_data["correlation"] = all_data["correlation"]/all_data["correlation"].max()
all_data["average"] = all_data["average"]/all_data["average"].max()
#all_data["vel_mag_corr"] = all_data["vel_mag_corr"]/all_data["vel_mag_corr"].max()

features = all_data[["correlation","average"]]
# data = np.array(np.c_[correlation,average])

## Unsupervised machine learning to find clusters in the 2 dimentional parameter space. 

kmeans = KMeans(n_clusters=3,max_iter=600)
kmeans_pred = kmeans.fit_predict(features)
all_data["kmeans"] = kmeans_pred

dbscan = DBSCAN(eps=0.03, min_samples=40)
dbscan_pred = dbscan.fit_predict(features)
all_data["DBSCAN"] = dbscan_pred

Agglo_cluster = AgglomerativeClustering(n_clusters=3,linkage="ward")
Agglo_pred = Agglo_cluster.fit_predict(features)
all_data["Agglo_cluster"] = Agglo_pred

Gaussian_mixture = GaussianMixture(n_components=3,covariance_type="tied")
Gaussian_pred = Gaussian_mixture.fit_predict(features)
all_data["Gaussian_mixture"] = Gaussian_pred

## plots

# fig0 = plt.axes(projection="3d")
x=all_data["correlation"]
y=all_data["average"]
z=all_data["vel_pas_corr"]
# img = fig0.scatter(x,y,z)
# plt.colorbar(img)
# fig0.set_xlabel("C(B,N)")
# fig0.set_ylabel("A(B,N)")
# fig0.set_zlabel("C(B,V)")
# fig0.set_title("Uncategorised Data")
# plt.show()


## First a plot containing all four of the algorithms
fig,axs = plt.subplots(2,2)
fig.suptitle("Time Window of "+time)
c = all_data["kmeans"]
axs[0,0].scatter(x,y,c=c,marker=".",cmap=plt.jet())
# axs[0,0].set_xlabel("C(B,N)")
axs[0,0].set_ylabel("A(B,N)")
axs[0,0].set_title("KMeans")

c = all_data["DBSCAN"]
axs[0,1].scatter(x,y,c=c,marker=".",cmap=plt.jet())
# axs[0,1].set_xlabel("C(B,N)")
# axs[0,1].set_ylabel("A(B,N)")
axs[0,1].set_title("DBSCAN")

c = all_data["Agglo_cluster"]
axs[1,0].scatter(x,y,c=c,marker=".",cmap=plt.jet())
axs[1,0].set_xlabel("C(B,N)")
axs[1,0].set_ylabel("A(B,N)")
axs[1,0].set_title("Agglomerative cluster")

c = all_data["Gaussian_mixture"]
axs[1,1].scatter(x,y,c=c,marker=".",cmap=plt.jet())
axs[1,1].set_xlabel("C(B,N)")
# axs[1,1].set_ylabel("A(B,N)")
axs[1,1].set_title("Gaussian mixture")

plt.show()


fig1 = plt.axes()
c = all_data["kmeans"]
img = fig1.scatter(x,y,c=c,cmap=plt.jet())
fig1.set_xlabel("C(B,N)")
fig1.set_ylabel("A(B,N)")
plt.title("KMeans"+time)
plt.show()
# plt.savefig(graph+"KMeans")
plt.clf()

fig2 = plt.axes()
c = all_data["DBSCAN"]
img = fig2.scatter(x,y,c=c,cmap=plt.jet())
fig2.set_xlabel("C(B,N)")
fig2.set_ylabel("A(B,N)")
plt.title("DBSCAN"+time)
# plt.savefig(graph+"DBSCAN")
plt.show()
plt.clf()

fig3 = plt.axes()
c = all_data["Agglo_cluster"]
img = fig3.scatter(x,y,c=c,cmap=plt.jet())
fig3.set_xlabel("C(B,N)")
fig3.set_ylabel("A(B,N)")
plt.title("Agglomerative cluster with "+time+" window")
# plt.savefig(graph+"Agglomerative Cluster")
plt.show()
plt.clf()

fig4 = plt.axes()
c = all_data["Gaussian_mixture"]
img = fig4.scatter(x,y,c=c,cmap=plt.jet())
fig4.set_xlabel("C(B,N)")
fig4.set_ylabel("A(B,N)")
plt.title("Gaussian mixture with "+time+" window")
# plt.savefig(graph+"Gaussian Mixture")
plt.show()
plt.clf()
x=all_data["correlation"]
y=all_data["average"]
z=all_data["vel_pas_corr"]
fig5 = plt.axes(projection="3d")
c = all_data["kmeans"]
img = fig5.scatter(x,y,z,marker=".",c=c,cmap=plt.jet())
fig5.set_xlabel("C(B,N)")
fig5.set_ylabel("A(B,N)")
fig5.set_zlabel("C(B,V)")
fig5.set_title("Kmeans with "+time+" window")
# plt.show()
plt.savefig(graph+"KMeans_vel")
plt.show()
plt.clf()

fig6 = plt.axes(projection="3d")
c = all_data["DBSCAN"]
img = fig6.scatter(x,y,z,marker=".",c=c,cmap=plt.jet())
fig6.set_xlabel("C(B,N)")
fig6.set_ylabel("A(B,N)")
fig6.set_zlabel("C(B,V)")
fig6.set_title("DBSCAN with "+time+" window")
plt.show()
plt.savefig(graph+"DBSCAN_vel")
plt.clf()

fig7 = plt.axes(projection="3d")
c = all_data["Agglo_cluster"]
img = fig7.scatter(x,y,z,marker=".",c=c,cmap=plt.jet())
fig7.set_xlabel("C(B,N)")
fig7.set_ylabel("A(B,N)")
fig7.set_zlabel("C(B,V)")
fig7.set_title("Aggolomerative Cluster with "+time+" window")
plt.savefig(graph+"Aggolo_cluster_vel")
plt.show()
plt.clf()

# fig8 = plt.axes(projection="3d")
# c = all_data["Gaussian_mixture"]
# img = fig8.scatter(x,y,z,marker=".",c=c,cmap=plt.jet())
# fig8.set_xlabel("C(B,N)")
# fig8.set_ylabel("A(B,N)")
# fig8.set_zlabel("C(B,V)")
# fig8.set_title("Gaussian Mixture with "+time+" window")
# plt.show()
# plt.savefig(graph+"Gaussian_Mixture_vel")
# plt.clf()


all_data["k_means"] = kmeans_pred
all_data["dbscan"] = dbscan_pred
all_data["Agglo"] = Agglo_pred
all_data["Gaussian"] = Gaussian_pred

df["k_means"] = all_data["k_means"]
df["k_means"] = df["k_means"].fillna(0)
df["dbscan"] = all_data["dbscan"]
df["dbscan"] = df["dbscan"] .fillna(0)
df["Agglo"] = all_data["Agglo"]
df["Agglo"] = df["Agglo"].fillna(0)
df["Gaussian"] = all_data["Gaussian"]
df["Gaussian"] = df["Gaussian"].fillna(0)
#df = df.drop(["mag_normalised","pas_normalised"], axis=1)
print(df)
df.to_csv(write_path)



# # # # ax.scatter3D(x,y,z)

# # # # #print(kmeans_pred==0,0)




# # # # fig = plt.figure()
# # # # ax = fig.add_subplot(111, projection = "3d")

# # # # ax = plt.axes(projection="3d")
# # # # z=all_data["correlation"]
# # # # x=all_data["average"]
# # # # y=all_data["vel_par_average"]
# # # # ax.scatter3D(x,y,z)
# # # # ax.set_xlabel("correlation")
# # # # ax.set_ylabel("average")
# # # # ax.set_zlabel("velocity")



# # # # plt.title("KMeans")
# # # # name = "KMeans.png"
# # # # #plt.savefig(graph_path+name)
# # # # plt.show()

# # # # plt.scatter(data[kmeans_pred==0,0], data[kmeans_pred==0,1], c='b')
# # # # plt.scatter(data[kmeans_pred==1,0], data[kmeans_pred==1,1], c='y')
# # # # plt.scatter()
# # # # plt.xlabel("Correlation")
# # # # plt.ylabel("Average")
# # # # plt.show()

# # # # # plt.scatter(data[dbscan_pred==0,0], data[dbscan_pred==0,1], c='b')
# # # # # plt.scatter(data[dbscan_pred==1,0], data[dbscan_pred==1,1], c='y')
# # # # # plt.xlabel("Correlation")
# # # # # plt.ylabel("Average")
# # # # # plt.title("DBScan")
# # # # # name = "DBScan.png"
# # # # # plt.savefig(graph_path+name)
# # # # # plt.show()

# # # plt.scatter(data[Agglo_pred==0,0], data[Agglo_pred==0,1], c='b')
# # # plt.scatter(data[Agglo_pred==1,0], data[Agglo_pred==1,1], c='y')
# # # plt.xlabel("Correlation")
# # # plt.ylabel("Average")
# # # plt.title("Agglo_cluster")
# # # name = "Agglo_cluster.png"
# # # plt.savefig(graph_path+name)
# # # plt.show()

# # # plt.scatter(data[Gaussian_pred==0,0], data[Gaussian_pred==0,1], c='b')
# # # plt.scatter(data[Gaussian_pred==1,0], data[Gaussian_pred==1,1], c='y')
# # # plt.xlabel("Correlation")
# # # plt.ylabel("Average")
# # # plt.title("Gaussian")
# # # name = "Gaussian.png" 
# # # plt.show()
# # # plt.savefig(graph_path+name)




# # # # plt.scatter(all_data["correlation"],all_data["average"], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
# # # # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
# # # # plt.show()


# # # # all_data["k_means_prediciton"] = kmeans_pred
# # # # #print(all_data["k_means_prediciton"])

