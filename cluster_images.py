import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Load the provided CSV file
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Load the pre-trained CNN model (ResNet50)
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Preprocessing function for the images
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Extract features from images
features = []
with torch.no_grad():  # No need to track gradients for feature extraction
    for img_path in df['Id']:
        print(img_path,"*************************")
        img_tensor = load_and_preprocess_image(img_path)
        img_features = model(img_tensor)
        features.append(img_features.squeeze().numpy())

# Convert features to a NumPy array
features = np.array(features)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Save the updated dataframe
output_file_path = 'clustered_data.csv'
df.to_csv(output_file_path, index=False)

# Display the first few rows of the resulting DataFrame
df.head()

