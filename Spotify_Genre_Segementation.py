import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/content/spotify dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Features for clustering
features = ['danceability', 'energy', 'tempo', 'liveness', 'valence', 
            'loudness', 'speechiness', 'acousticness', 'instrumentalness']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.show()

# Choose the number of clusters (k) based on the elbow plot
k = 5  # Example: Choosing k=5

# Fit KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='danceability', y='energy', hue='cluster', palette='viridis', s=100)
plt.title('Spotify Genre Segmentation')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Optional: Display the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers (original scale):")
print(cluster_centers)
