import numpy as np

# Load the .npy file
file_path = "/Users/tienphat/Downloads/cluster_labels_en_cosine (1) (1).npy"
data = np.load(file_path)

# Print the dimensions of the loaded array
print("Dimensions of the array:", data.shape)