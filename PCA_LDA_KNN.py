import matplotlib.pyplot as plt  # Used for plotting various visualizations
import numpy as np  # Used for scientific computing and array operations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Used for LDA dimensionality reduction
from sklearn.model_selection import train_test_split  # Used for splitting the dataset into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # Used for K-Nearest Neighbors classifier
from sklearn.metrics import accuracy_score  # Used for calculating classification accuracy

label = np.load("animal_labels.npy")  # Load label information, containing the class of each sample
resnet_set = np.load("animal_resnet_features.npy")  # Load feature dataset for LDA dimensionality reduction
classifier_set = np.load("animal_raw_features.npy")  # Load feature dataset for PCA dimensionality reduction

# Split the dataset into training and testing sets
resnet_train, resnet_test, label_train, label_test = train_test_split(resnet_set, label, test_size=0.3, random_state=42)  # Split ResNet feature dataset and labels
classifier_train, classifier_test = train_test_split(classifier_set, test_size=0.3, random_state=42)  # Split classifier feature dataset

# Reshape the feature datasets to make them suitable for the dimensionality reduction algorithm input format
resnet_train_reshape = resnet_train.reshape(resnet_train.shape[0], 8 * 1024)  # Reshape ResNet training features to be suitable for LDA model input. The value '8 * 1024' is chosen based on the feature dimensions of the ResNet output.  # Reshape ResNet training features to be suitable for LDA model input
resnet_test_reshape = resnet_test.reshape(resnet_test.shape[0], 8 * 1024)  # Reshape ResNet testing features to be suitable for LDA model input
classifier_train_reshape = classifier_train.reshape(classifier_train.shape[0], 1 * 1024)  # Reshape classifier training features to be suitable for PCA model input. The value '1 * 1024' is chosen based on the feature dimensions of the raw data.  # Reshape classifier training features to be suitable for PCA model input
classifier_test_reshape = classifier_test.reshape(classifier_test.shape[0], 1 * 1024)  # Reshape classifier testing features to be suitable for PCA model input

# Calculate the mean of the training set and perform mean normalization
mean_vector = np.mean(classifier_train_reshape, axis=0)  # Calculate the mean of each feature along the columns to standardize the dataset for PCA.  # Calculate the mean of each feature
classifier_train_centered = classifier_train_reshape - mean_vector  # Perform mean normalization on the training set
classifier_test_centered = classifier_test_reshape - mean_vector  # Perform mean normalization on the testing set

# Calculate the covariance matrix
cov_matrix = np.cov(classifier_train_centered, rowvar=False)  # Calculate the covariance matrix

# Perform eigenvalue decomposition to get eigenvalues and eigenvectors
eig_values, eig_vectors = np.linalg.eigh(cov_matrix)  # Calculate the eigenvalues and eigenvectors of the covariance matrix

# Sort the eigenvalues in descending order
sorted_indices = np.argsort(eig_values)[::-1]  # Sort eigenvalues in descending order to prioritize the most significant components for dimensionality reduction.  # Get the indices of eigenvalues sorted in descending order
sorted_eig_values = eig_values[sorted_indices]  # Sort eigenvalues according to the indices
sorted_eig_vectors = eig_vectors[:, sorted_indices]  # Sort eigenvectors according to the indices

# Select the top 2 eigenvectors for dimensionality reduction
pca_2d_components = sorted_eig_vectors[:, :2]  # Select the top 2 eigenvectors
X_pca_2d_train = classifier_train_centered @ pca_2d_components  # Perform dimensionality reduction on the training set
X_pca_2d_test = classifier_test_centered @ pca_2d_components  # Perform dimensionality reduction on the testing set

# Select the top 3 eigenvectors for dimensionality reduction
pca_3d_components = sorted_eig_vectors[:, :3]  # Select the top 3 eigenvectors
X_pca_3d_train = classifier_train_centered @ pca_3d_components  # Perform dimensionality reduction on the training set
X_pca_3d_test = classifier_test_centered @ pca_3d_components  # Perform dimensionality reduction on the testing set

# Load the LDA model and train it to reduce data to 2D
model_lda_2d = LinearDiscriminantAnalysis(n_components=2)  # Initialize the LDA model to reduce to 2D. The number of components is chosen to visualize the data in two dimensions.  # Initialize the LDA model to reduce to 2D
X_lda_2d_train = model_lda_2d.fit_transform(resnet_train_reshape, label_train)  # Use LDA model to reduce dimensionality of training features
X_lda_2d_test = model_lda_2d.transform(resnet_test_reshape)  # Reduce dimensionality of testing features

# Construct data for 3D visualization by adding a zero z-axis for pseudo 3D display
z_value = np.zeros(X_lda_2d_test.shape[0])  # Create a zero array with the same number of rows as LDA test set results, used as z-axis values

# Calculate reconstruction error (reconstruction error rate)
# For PCA dimensionality reduction results, reconstruct using the dimensionality reduction matrix
reconstructed_pca_2d_train = X_pca_2d_train @ pca_2d_components.T + mean_vector  # Reconstruct 2D reduced training data
reconstructed_pca_2d_test = X_pca_2d_test @ pca_2d_components.T + mean_vector  # Reconstruct 2D reduced testing data
reconstructed_pca_3d_train = X_pca_3d_train @ pca_3d_components.T + mean_vector  # Reconstruct 3D reduced training data
reconstructed_pca_3d_test = X_pca_3d_test @ pca_3d_components.T + mean_vector  # Reconstruct 3D reduced testing data

# Calculate reconstruction error rate for 2D and 3D dimensionality reduction using the eigenvalue calculation method
pca_2d_train_error_rate = np.sum(sorted_eig_values[2:]) / np.sum(sorted_eig_values) * 100  # Reconstruction error using top 2 eigenvectors
pca_2d_test_error_rate = pca_2d_train_error_rate  # Testing set error rate is the same as training set
pca_3d_train_error_rate = np.sum(sorted_eig_values[3:]) / np.sum(sorted_eig_values) * 100  # Reconstruction error using top 3 eigenvectors
pca_3d_test_error_rate = pca_3d_train_error_rate  # Testing set error rate is the same as training set

# Use K-Nearest Neighbors classifier for classification tasks
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Initialize KNN classifier with 3 neighbors

# Classify data reduced by PCA (2D)
knn_classifier.fit(X_pca_2d_train, label_train)  # Train the classifier using the training set
pca_2d_pred = knn_classifier.predict(X_pca_2d_test)  # Predict the testing set
pca_2d_classification_accuracy = accuracy_score(label_test, pca_2d_pred) * 100  # Calculate classification accuracy

# Classify data reduced by PCA (3D)
knn_classifier.fit(X_pca_3d_train, label_train)  # Train the classifier using the training set
pca_3d_pred = knn_classifier.predict(X_pca_3d_test)  # Predict the testing set
pca_3d_classification_accuracy = accuracy_score(label_test, pca_3d_pred) * 100  # Calculate classification accuracy

# Classify data reduced by LDA (2D)
knn_classifier.fit(X_lda_2d_train, label_train)  # Train the classifier using the training set
lda_2d_pred = knn_classifier.predict(X_lda_2d_test)  # Predict the testing set
lda_2d_classification_accuracy = accuracy_score(label_test, lda_2d_pred) * 100  # Calculate classification accuracy

# LDA does not support inverse_transform, so the error calculation method is different; calculate classification error rate
lda_2d_error_rate = 100 - lda_2d_classification_accuracy  # Calculate classification error rate since LDA does not support reconstruction-based error calculation like PCA.  # Calculate classification error rate

# Plot comparison of reconstruction error rates for training and testing sets
train_error_rates = [pca_2d_train_error_rate, pca_3d_train_error_rate]  # Store PCA 2D and PCA 3D training set error rates
test_error_rates = [pca_2d_test_error_rate, pca_3d_test_error_rate]  # Store PCA 2D and PCA 3D testing set error rates
methods = ['PCA (2D)', 'PCA (3D)']  # Corresponding methods

x = np.arange(len(methods))  # X-axis positions
width = 0.25  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6), dpi=80)  # Initialize canvas
bar1 = ax.bar(x - width/2, train_error_rates, width, label='Train Error Rate', color='skyblue')  # Plot training set error rate bar chart
bar2 = ax.bar(x + width/2, test_error_rates, width, label='Test Error Rate', color='orange')  # Plot testing set error rate bar chart

ax.set_xlabel('Dimensionality Reduction Method')  # Set x-axis label
ax.set_ylabel('Error Rate (%)')  # Set y-axis label
ax.set_title('Train vs Test Error Rates for PCA Dimensionality Reduction')  # Set title
ax.set_xticks(x)
ax.set_xticklabels(methods)  # Set x-axis tick labels
ax.legend()  # Display legend
ax.grid(axis='y')  # Display y-axis grid

plt.ylim(0, max(max(train_error_rates), max(test_error_rates)) + 5)  # Set y-axis range
plt.tight_layout()  # Automatically adjust subplot parameters to fill the canvas
plt.show()  # Show plot

# Plot error rate chart
error_rates = [pca_2d_test_error_rate, pca_3d_test_error_rate, lda_2d_error_rate]  # Store PCA 2D, PCA 3D, and LDA 2D testing set error rates
classification_accuracies = [pca_2d_classification_accuracy, pca_3d_classification_accuracy, lda_2d_classification_accuracy]  # Store PCA 2D, PCA 3D, and LDA 2D classification accuracies
methods = ['PCA (2D)', 'PCA (3D)', 'LDA (2D)']  # Corresponding methods

plt.figure(figsize=(10, 6), dpi=80)  # Initialize canvas
plt.bar(methods, error_rates, color=['skyblue', 'orange', 'green'], alpha=0.7, label='Error Rate (%)')  # Plot error rate bar chart
plt.bar(methods, classification_accuracies, color=['blue', 'red', 'purple'], alpha=0.7, label='Classification Accuracy (%)', bottom=error_rates)  # Plot classification accuracy bar chart
plt.xlabel('Dimensionality Reduction Method')  # Set x-axis label
plt.ylabel('Percentage (%)')  # Set y-axis label
plt.title('Error Rates and Classification Accuracies for Different Dimensionality Reduction Methods')  # Set title
plt.ylim(0, max(error_rates) + max(classification_accuracies) + 5)  # Set y-axis range
plt.legend()  # Display legend
plt.grid(axis='y')  # Display y-axis grid
plt.show()  # Show plot

# Plot 2D scatter plot (PCA)
labels = [0, 1, 2, 3, 4, 5]  # Define label categories, with 6 classes in total
Colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']  # Define color for each label
label_express = ['Dog', 'Monkey', 'Cat', 'Human', 'Horse', 'Cow']  # Define label meanings

plt.figure(figsize=(10, 8), dpi=100)  # Initialize canvas, set size and resolution
plt.title(f'Transformed Samples via PCA (2D)Error Rate: {pca_2d_test_error_rate:.2f}%')  # Set title with error rate information
plt.xlabel('Principal Component 1')  # Set x-axis label
plt.ylabel('Principal Component 2')  # Set y-axis label
plt.xlim([-70, 70])  # Set x-axis range for better data display. These limits are chosen based on prior analysis to ensure all data points are visible.  # Set x-axis range for better data display
plt.ylim([-50, 70])  # Set y-axis range for better data display. These limits are chosen based on prior analysis to ensure all data points are visible.  # Set y-axis range for better data display

# Iterate through each label and plot data points for the corresponding category
for tlabel in labels:
    x_pca_data = X_pca_2d_test[label_test == tlabel, 0]  # Get x-coordinates of data points for this label
    y_pca_data = X_pca_2d_test[label_test == tlabel, 1]  # Get y-coordinates of data points for this label
    plt.scatter(x=x_pca_data, y=y_pca_data, s=40, c=Colors[tlabel], label=label_express[tlabel], alpha=0.7, edgecolors='w', linewidth=0.5)  # Plot scatter points with transparency and edge

plt.legend(loc="best", fontsize=9)  # Set legend position and font size
plt.grid(True, linestyle='--', alpha=0.6)  # Display grid with dashed lines and transparency
plt.tight_layout()  # Automatically adjust subplot parameters to fill the canvas
plt.show()  # Show plot

# Plot 2D scatter plot (LDA)
plt.figure(figsize=(8, 6), dpi=80)  # Initialize canvas
plt.title(f'Transformed samples via LDA (2D)Error Rate: {lda_2d_error_rate:.2f}%')  # Set title with error rate information
plt.xlabel('x_values')  # Set x-axis label
plt.ylabel('y_values')  # Set y-axis label

# Iterate through each label and plot data points for the corresponding category
for tlabel in labels:
    x_lda_data = X_lda_2d_test[label_test == tlabel, 0]  # Get x-coordinates of data points for this label
    y_lda_data = X_lda_2d_test[label_test == tlabel, 1]  # Get y-coordinates of data points for this label
    plt.scatter(x=x_lda_data, y=y_lda_data, s=20, c=Colors[tlabel], label=label_express[tlabel])  # Plot scatter points

plt.legend(loc="upper right")  # Set legend position
plt.grid()  # Display grid
plt.show()  # Show plot

# Plot 3D scatter plot (PCA)
fig = plt.figure(figsize=(8, 6), dpi=80)  # Initialize 3D canvas
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
ax.set_title(f'Transformed samples via PCA (3D)Error Rate: {pca_3d_test_error_rate:.2f}%')  # Set title with error rate information
ax.set_xlabel('x_value')  # Set x-axis label
ax.set_ylabel('y_value')  # Set y-axis label
ax.set_zlabel('z_value')  # Set z-axis label

# Iterate through each label and plot data points for the corresponding category
for tlabel in labels:
    x_pca_data = X_pca_3d_test[label_test == tlabel, 0]  # Get x-coordinates of data points for this label
    y_pca_data = X_pca_3d_test[label_test == tlabel, 1]  # Get y-coordinates of data points for this label
    z_pca_data = X_pca_3d_test[label_test == tlabel, 2]  # Get z-coordinates of data points for this label
    ax.scatter(xs=x_pca_data, ys=y_pca_data, zs=z_pca_data, s=20, c=Colors[tlabel], label=label_express[tlabel])  # Plot 3D scatter points. The marker size and color are chosen to ensure clear visualization of each class.  # Plot 3D scatter points

plt.legend(loc="upper right")  # Set legend position
plt.show()  # Show plot

# Plot 3D scatter plot (LDA pseudo 3D)
fig = plt.figure(figsize=(8, 6), dpi=80)  # Initialize 3D canvas
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
ax.set_title('Transformed samples via LDA (pseudo 3D)')  # Set title
ax.set_xlabel('x_value')  # Set x-axis label
ax.set_ylabel('y_value')  # Set y-axis label
ax.set_zlabel('z_value')  # Set z-axis label

# Iterate through each label and plot data points for the corresponding category
for tlabel in labels:
    x_lda_data = X_lda_2d_test[label_test == tlabel, 0]  # Get x-coordinates of data points for this label
    y_lda_data = X_lda_2d_test[label_test == tlabel, 1]  # Get y-coordinates of data points for this label
    z_lda_data = z_value[label_test == tlabel]  # Use z_value to fill z-axis, making it zero to form pseudo 3D plot
    ax.scatter(xs=x_lda_data, ys=y_lda_data, zs=z_value[label_test == tlabel], s=20, c=Colors[tlabel], label=label_express[tlabel])  # Plot 3D scatter points

plt.legend(loc="upper right")  # Set legend position
plt.show()  # Show plot