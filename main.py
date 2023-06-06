import pandas as pd

from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import re
import nltk
from nltk.corpus import stopwords

# Get the list of stopwords

nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))

df = pd.read_csv("data/training_data.tsv.gz", sep="\t", header=None).head(1000)

df.rename(columns={0: 'index', 1: 'title', 2: 'text', 3: 'labels'}, inplace=True)
df.drop('index', axis=1, inplace=True)

X = df.drop('labels', axis=1)
X = X.applymap(lambda x: re.sub(r'<.*?>|[^\w\s]', '', x.lower())).applymap(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords_list]))
y = df['labels'].str.get_dummies(',')

X = X["title"] + X["text"]

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
X = vectorizer.fit_transform(X)

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

num_clusters = 5

# Initialize the KNN classifier
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_train)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_variation_b.ipynb

closest_centroids = kmeans.predict(X_test)

centroid_labels = []
for i in range(num_clusters):
    cluster_data = X_train[labels == i]
    cluster_label = ...  # Determine the label for the cluster based on the data
    centroid_labels.append(cluster_label)


# Plot known data and centroids
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap='viridis', alpha=0.5, label='Known Data')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids')
for i, label in enumerate(centroid_labels):
    plt.annotate(label, (centroids[i, 0], centroids[i, 1]), color='red')
plt.legend()
plt.title('Known Data and Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()