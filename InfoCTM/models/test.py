import numpy as np

# Load the .npy file
file_path = '/Users/tienphat/Downloads/InfoCTM/data/Amazon_Review/cluster_labels_cn_cosine.npy'
data = np.load(file_path)

# Print the dimensions of the loaded data
print("Number of unique values in the loaded data:", np.unique(data).size)