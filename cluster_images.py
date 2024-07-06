# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import os
# import numpy as np
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models


# file_path = 'dataset.csv'
# df = pd.read_csv(file_path)


# model = models.resnet50(pretrained=True)
# model.eval() 


# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# def load_and_preprocess_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     img = preprocess(img)
#     img = img.unsqueeze(0)  # Add batch dimension
#     return img


# features = []
# with torch.no_grad():  
#     for img_path in df['Id']:       
#         img_tensor = load_and_preprocess_image(img_path)
#         img_features = model(img_tensor)
#         features.append(img_features.squeeze().numpy())


# features = np.array(features)


# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)


# kmeans = KMeans(n_clusters=4, random_state=42)
# df['Cluster'] = kmeans.fit_predict(features_scaled)


# output_file_path = 'clustered_data.csv'
# df.to_csv(output_file_path, index=False)


# df.head()
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

file_path = 'datasheets/data.csv'
df = pd.read_csv(file_path)

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure all images are the same size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img)
    img = img.view(-1)  # Flatten the image to 1D
    return img

features = []
for img_path in df['Id']:
    img_tensor = load_and_preprocess_image(img_path)
    features.append(img_tensor.numpy())

features = np.array(features)

# Use KNN for feature extraction
knn = NearestNeighbors(n_neighbors=5)
knn.fit(features)

# Transform features using KNN
features_transformed = knn.kneighbors(features, return_distance=False)

# Flatten the features for clustering
features_flattened = features_transformed.reshape(features_transformed.shape[0], -1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_flattened)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

output_file_path = 'datasheets/clustered_data.csv'
df.to_csv(output_file_path, index=False)

df.head()
