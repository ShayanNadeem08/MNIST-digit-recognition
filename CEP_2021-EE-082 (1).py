#!/usr/bin/env python
# coding: utf-8

# **Trains SVM(Feature Pixel value) on MNIST with varying training examples, also evaluating confusion matrices.**
# 

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
start_time = time.time()
# Load MNIST dataset
mnist_data = np.load('mnist-data.npz')

# Extract features and labels
training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

# Reshape images to flat format (28x28 to 784)
training_data_flat = training_data.reshape(training_data.shape[0], -1)

validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)                    

num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000]

train_accuracies = []
val_accuracies = []
conf_matrices = []

for num_examples in num_training_examples:
    
    subset_train_data = train_data[:num_examples]
    subset_train_labels = train_labels[:num_examples]
    
    svm_model = svm.SVC(kernel='poly', degree=2)
    svm_model.fit(subset_train_data, subset_train_labels)

    train_predictions = svm_model.predict(subset_train_data)
    train_accuracy = accuracy_score(subset_train_labels, train_predictions)
    train_accuracies.append(train_accuracy)

    val_predictions = svm_model.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)

    conf_matrix = confusion_matrix(val_labels, val_predictions)
    conf_matrices.append(conf_matrix)

plt.figure(figsize=(10, 6))
plt.plot(num_training_examples, train_accuracies, label='Training Accuracy')
plt.plot(num_training_examples, val_accuracies, label='Validation Accuracy')
plt.title('MNIST Linear SVM Training and Validation Metrics')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 10))
for i, conf_matrix in enumerate(conf_matrices):
    plt.subplot(3, 3, i + 1)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g', cbar=False)
    plt.title(f"Confusion Matrix (Num Examples: {num_training_examples[i]})")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
plt.tight_layout()
plt.show()

end_time = time.time()
total_time= end_time-start_time
print("The time for execution is: ",total_time,"s")


# **Trains SVM(Feature: HOG) on MNIST with varying training examples, also evaluating confusion matrices.**

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure

mnist_data = np.load('mnist-data.npz')

training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

training_data_flat = training_data.reshape(training_data.shape[0], -1)

validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)

num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000]

train_accuracies = []
val_accuracies = []
conf_matrices = []

for num_examples in num_training_examples:

    subset_train_data = train_data[:num_examples]
    subset_train_labels = train_labels[:num_examples]


    hog_features = []
    for image in subset_train_data:
    
        fd, hog_image = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
        hog_features.append(fd)


    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(hog_features, subset_train_labels)


    train_predictions = svm_model.predict(hog_features)
    train_accuracy = accuracy_score(subset_train_labels, train_predictions)
    train_accuracies.append(train_accuracy)


    val_hog_features = []
    for image in val_data:
        fd, hog_image = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
        val_hog_features.append(fd)

    val_predictions = svm_model.predict(val_hog_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)


    conf_matrix = confusion_matrix(val_labels, val_predictions)
    conf_matrices.append(conf_matrix)

plt.figure(figsize=(10, 6))
plt.plot(num_training_examples, train_accuracies, label='Training Accuracy')
plt.plot(num_training_examples, val_accuracies, label='Validation Accuracy')
plt.title('MNIST SVM with HOG Features Training and Validation Metrics')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 10))
for i, conf_matrix in enumerate(conf_matrices):
    plt.subplot(3, 3, i + 1)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g', cbar=False)
    plt.title(f"Confusion Matrix (Num Examples: {num_training_examples[i]})")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
plt.tight_layout()
plt.show()


# **MNIST Random Forest with HOG Features**

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import time
start_time = time.time()
mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']


training_data_flat = training_data.reshape(training_data.shape[0], -1)


validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)


num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000]


train_accuracies = []
val_accuracies = []


for num_examples in num_training_examples:
   
    subset_train_data = train_data[:num_examples]
    subset_train_labels = train_labels[:num_examples]

    
    hog_features_train = []
    for image in subset_train_data:
        fd = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(1, 1), visualize=False, multichannel=False)
        hog_features_train.append(fd)


    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    
    rf_model.fit(hog_features_train, subset_train_labels)
    

  
    train_predictions = rf_model.predict(hog_features_train)
    train_accuracy = accuracy_score(subset_train_labels, train_predictions)
    train_accuracies.append(train_accuracy)

    
    hog_features_val = []
    for image in val_data:
        fd = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(1, 1), visualize=False, multichannel=False)
        hog_features_val.append(fd)

    
    val_predictions = rf_model.predict(hog_features_val)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)


plt.figure(figsize=(10, 6))
plt.plot(num_training_examples, train_accuracies, label='Training Accuracy')
plt.plot(num_training_examples, val_accuracies, label='Validation Accuracy')
plt.title('MNIST Random Forest with HOG Features Training and Validation Metrics')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Execution time for the code is: {execution_time:.2f} seconds")


# In[5]:


print("The max validation accuracy = ",max(val_accuracies))


# **Random forest on mnist with PCA as features**

# In[10]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
start_time = time.time()


mnist_data = np.load('mnist-data.npz')

training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

training_data_flat = training_data.reshape(training_data.shape[0], -1)

validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)

n_components =300  # Number of principal components to retain
pca = PCA(n_components=n_components)
pca.fit(train_data)

train_data_pca = pca.transform(train_data)
val_data_pca = pca.transform(val_data)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_data_pca, train_labels)

train_predictions = rf_model.predict(train_data_pca)
train_accuracy = accuracy_score(train_labels, train_predictions)
print("Training Accuracy:", train_accuracy)

val_predictions = rf_model.predict(val_data_pca)
val_accuracy = accuracy_score(val_labels, val_predictions)
print("Validation Accuracy:", val_accuracy)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time for the code is: {execution_time:.2f} seconds")


# In[18]:


plt.figure(figsize=(12, 6))
for i in range(num_images_to_plot):
    # Original image
    plt.subplot(2, num_images_to_plot, i + 1)
    plt.imshow(training_data_flat[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # PCA Reduced image
    plt.subplot(2, num_images_to_plot, num_images_to_plot + i + 1)
    plt.imshow(reduced_images[i].reshape(10, 30), cmap='gray')
    plt.title('PCA Reduced')
    plt.axis('off')
    
    
plt.tight_layout()
plt.show()


# **Trying several regression techniques**

# In[24]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

# Reshape images to flat format (28x28 to 784)
training_data_flat = training_data.reshape(training_data.shape[0], -1)


validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)

# Train Logistic Regression model with OvR strategy
logreg_model = LogisticRegression(multi_class='ovr', max_iter=1000)
logreg_model.fit(train_data, train_labels)

# Predict on training set
train_predictions = logreg_model.predict(train_data)
train_accuracy = accuracy_score(train_labels, train_predictions)
print("Training Accuracy:", train_accuracy)

# Predict on validation set
val_predictions = logreg_model.predict(val_data)
val_accuracy = accuracy_score(val_labels, val_predictions)
print("Validation Accuracy:", val_accuracy)


# In[25]:


# Train Logistic Regression model with softmax regression
logreg_softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logreg_softmax_model.fit(train_data, train_labels)

# Predict on training set
train_predictions_softmax = logreg_softmax_model.predict(train_data)
train_accuracy_softmax = accuracy_score(train_labels, train_predictions_softmax)
print("Training Accuracy (Softmax Regression):", train_accuracy_softmax)

# Predict on validation set
val_predictions_softmax = logreg_softmax_model.predict(val_data)
val_accuracy_softmax = accuracy_score(val_labels, val_predictions_softmax)
print("Validation Accuracy (Softmax Regression):", val_accuracy_softmax)


# **Applying LDA on mnist**

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']


training_data_flat = training_data.reshape(training_data.shape[0], -1)


validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)

# Apply Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis(n_components=2)
train_data_lda = lda.fit_transform(train_data, train_labels)
val_data_lda = lda.transform(val_data)

# Visualize original and reduced images
num_images_to_plot = 5
plt.figure(figsize=(12, 6))
for i in range(num_images_to_plot):
    # Original image
    plt.subplot(2, num_images_to_plot, i + 1)
    plt.imshow(train_data[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # LDA Reduced image
    plt.subplot(2, num_images_to_plot, num_images_to_plot + i + 1)
    plt.scatter(train_data_lda[:, 0], train_data_lda[:, 1], c=train_labels, cmap='viridis')
    plt.title('LDA Reduced')
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.colorbar()

plt.tight_layout()
plt.show()

train_accuracy = lda.score(train_data, train_labels)
print("Training Accuracy:", train_accuracy)


val_accuracy = lda.score(val_data, val_labels)
print("Validation Accuracy:", val_accuracy)


# **Naive Baye's classifier using raw pixel value as feature**

# In[7]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']


training_data_flat = training_data.reshape(training_data.shape[0], -1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(training_data_flat, training_labels, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on training and validation set
train_predictions = nb_classifier.predict(X_train)
val_predictions = nb_classifier.predict(X_val)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)

print("Training Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)


# **KNN with raw pixel value as features**

# In[12]:


gimport numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import time

start_time = time.time()

g
mnist_data = np.load('mnist-data.npz')

g
training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']
g
training_data_flat = training_data.reshape(training_data.shape[0], -1)

g
validation_size = 10000
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data_flat, training_labels, test_size=validation_size, random_state=42
)

g
num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000]

# Initialize lists to store training and validation accuracies
train_accuracies = []
val_accuracies = []
conf_matrices = []

# Train KNN model for each specified number of training examples
for num_examples in num_training_examples:
    # Select a subset of training data and labels
    subset_train_data = train_data[:num_examples]
    subset_train_labels = train_labels[:num_examples]

    # Train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    knn_model.fit(subset_train_data, subset_train_labels)

    # Predict on training set
    train_predictions = knn_model.predict(subset_train_data)
    train_accuracy = accuracy_score(subset_train_labels, train_predictions)
    train_accuracies.append(train_accuracy)

    # Predict on validation set
    val_predictions = knn_model.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(val_labels, val_predictions)
    conf_matrices.append(conf_matrix)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_training_examples, train_accuracies, label='Training Accuracy')
plt.plot(num_training_examples, val_accuracies, label='Validation Accuracy')
plt.title('MNIST KNN Training and Validation Metrics')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

print("Maximum Training accuracies =",max(train_accuracies))
print("Maximum Validation accuracies =",max(val_accuracies))


end_time = time.time()
total_time = end_time - start_time
print("The time for execution is:", total_time, "seconds")


# **Reducing dimensions from 784 to 2D using PCA**

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the scatter plot
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=str(i), alpha=0.5)
plt.title("Scatter Plot of MNIST Data (2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Digit")
plt.grid(True)
plt.show()


# **Quadratic SVM with images to patches as features**

# In[15]:


import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d

mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']


training_data_flat = training_data.reshape(training_data.shape[0], -1)

# Define patch size
patch_size = (4, 4)

# Extract patches from images
patches = []
for image in training_data:
    image = image.reshape(28, 28)  # Reshape image to original dimensions
    image_patches = extract_patches_2d(image, patch_size)
    patches.append(image_patches.reshape(-1, patch_size[0] * patch_size[1]))
patches = np.array(patches)

# Flatten patches and concatenate to get feature vector
patch_features = patches.reshape(patches.shape[0], -1)

# Set aside 10,000 training images as a validation set
validation_size = 10000
train_patches, val_patches, train_labels, val_labels = train_test_split(
    patch_features, training_labels, test_size=validation_size, random_state=42
)

# Train SVM model
svm_model = svm.SVC(kernel='poly', degree=2)
svm_model.fit(train_patches, train_labels)

# Predict on validation set
val_predictions = svm_model.predict(val_patches)
val_accuracy = accuracy_score(val_labels, val_predictions)
print("Validation Accuracy:", val_accuracy)


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

# Flatten images to use pixel values as featurs
pixel_features = training_data.reshape(training_data.shape[0], -1)

# Extract HOG features
hog_features = np.array([hog(image.reshape(28, 28), pixels_per_cell=(2, 2), cells_per_block=(1, 1)) for image in training_data])

# Define different models
models = {
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}


num_training_examples = [100, 1000, 5000, 10000]

# Initialize lists to store validation scores
validation_scores = {model_name: [] for model_name in models}

# Train and evaluate models for different numbers of training examples
for num_examples in num_training_examples:
    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        pixel_features[:num_examples], training_labels[:num_examples], test_size=0.2, random_state=42
    )

    for model_name, model in models.items():
        # Train the model
        model.fit(train_data, train_labels)
        
        # Predict on validation set
        val_predictions = model.predict(val_data)
        
        # Calculate accuracy and store validation score
        val_accuracy = accuracy_score(val_labels, val_predictions)
        validation_scores[model_name].append(val_accuracy)

# Plotting
plt.figure(figsize=(10, 6))
for model_name, scores in validation_scores.items():
    plt.plot(num_training_examples, scores, label=model_name)

plt.title('Validation Scores of Different Models')
plt.xlabel('Number of Training Examples')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.show()


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time
mnist_data = np.load('mnist-data.npz')


training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

# Flatten images to use pixel values as features
pixel_features = training_data.reshape(training_data.shape[0], -1)

# Extract HOG features
hog_features = np.array([hog(image.reshape(28, 28), pixels_per_cell=(2, 2), cells_per_block=(1, 1)) for image in training_data])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    pixel_features, training_labels, test_size=0.2, random_state=42
)

# Define different models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Linear SVM': SVC(kernel='linear'),
    'Quadratic SVM': SVC(kernel='poly', degree=2),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Define number of training examples
num_training_examples = [100, 1000, 5000, 10000]

# Initialize lists to store validation scores and execution times
validation_scores = {model_name: [] for model_name in models}
execution_times = {model_name: [] for model_name in models}

# Train and evaluate models for different numbers of training examples
for num_examples in num_training_examples:
    for model_name, model in models.items():
        start_time = time.time()
        # Train the model
        model.fit(X_train[:num_examples], y_train[:num_examples])
        
        # Predict on validation set
        val_predictions = model.predict(X_val)
        end_time = time.time()

        # Calculate accuracy and store validation score
        val_accuracy = accuracy_score(y_val, val_predictions)
        validation_scores[model_name].append(val_accuracy)

        # Calculate execution time and store
        execution_times[model_name].append(end_time - start_time)

# Plotting validation scores
plt.figure(figsize=(12, 8))
for model_name, scores in validation_scores.items():
    plt.plot(num_training_examples, scores, label=model_name)

plt.title('Validation Scores of Different Models')
plt.xlabel('Number of Training Examples')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.show()


# In[21]:


plt.figure(figsize=(12, 8))
for model_name, time_taken in execution_times.items():
    plt.plot(num_training_examples, time_taken, marker='o', label=model_name)

plt.title('Execution Time of Different Models')
plt.xlabel('Number of Training Examples')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.grid()
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time

# Load MNIST dataset
mnist_data = np.load('mnist-data.npz')

# Extract features and labels
training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

# Extract HOG features
hog_features = np.array([hog(image.reshape(28, 28), pixels_per_cell=(2, 2), cells_per_block=(1, 1)) for image in training_data])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    hog_features, training_labels, test_size=0.2, random_state=42
)

# Define different models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Linear SVM': SVC(kernel='linear'),
    'Quadratic SVM': SVC(kernel='poly', degree=2),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Define number of training examples
num_training_examples = [100, 1000, 5000, 10000]

# Initialize lists to store validation scores and execution times
validation_scores = {model_name: [] for model_name in models}
execution_times = {model_name: [] for model_name in models}

# Train and evaluate models for different numbers of training examples
for num_examples in num_training_examples:
    for model_name, model in models.items():
        start_time = time.time()
        # Train the model
        model.fit(X_train[:num_examples], y_train[:num_examples])
        
        # Predict on validation set
        val_predictions = model.predict(X_val)
        end_time = time.time()

        # Calculate accuracy and store validation score
        val_accuracy = accuracy_score(y_val, val_predictions)
        validation_scores[model_name].append(val_accuracy)

        # Calculate execution time and store
        execution_times[model_name].append(end_time - start_time)

# Plotting validation scores
plt.figure(figsize=(12, 8))
for model_name, scores in validation_scores.items():
    plt.plot(num_training_examples, scores, label=model_name)

plt.title('Validation Scores of Different Models (Using HOG Features)')
plt.xlabel('Number of Training Examples')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plotting execution times
plt.figure(figsize=(12, 8))
for model_name, times in execution_times.items():
    plt.plot(num_training_examples, times, label=model_name)

plt.title('Execution Time of Different Models (Using HOG Features)')
plt.xlabel('Number of Training Examples')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA

# Load MNIST dataset
mnist_data = np.load('mnist-data.npz')

# Extract features and labels
training_data = mnist_data['training_data']
training_labels = mnist_data['training_labels']

# Reshape images to flat format (28x28 to 784)
training_data_flat = training_data.reshape(training_data.shape[0], -1)

# Calculate mean of raw pixel values for each class
mean_raw_pixel_value = [np.mean(training_data_flat[training_labels == i]) for i in range(10)]

# Calculate mean of HOG features for each class
hog_features = np.array([hog(image.reshape(28, 28), pixels_per_cell=(2, 2), cells_per_block=(1, 1)) for image in training_data])
mean_hog_features = [np.mean(hog_features[training_labels == i]) for i in range(10)]

# Apply PCA to reduce dimensionality
pca = PCA(n_components=128)
training_data_pca = pca.fit_transform(training_data_flat)
mean_pca_features = [np.mean(training_data_pca[training_labels == i]) for i in range(10)]

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(10), mean_raw_pixel_value, label='Raw Pixel Value')
plt.scatter(range(10), mean_hog_features, label='HOG')
plt.scatter(range(10), mean_pca_features, label='PCA')
plt.xlabel('Class Label')
plt.ylabel('Mean of Features')
plt.title('Mean of Features by Class')
plt.xticks(range(10))
plt.legend()
plt.grid()
plt.show()


# In[ ]:




