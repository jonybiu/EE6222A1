import numpy as np
input_data_a = np.load(r"animal_raw_features.npy")
input_data_b = np.load(r"animal_resnet_features.npy")

print(input_data_a.shape) #输出其形状
print(input_data_b.shape) #输出其形状