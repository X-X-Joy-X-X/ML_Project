import numpy as np
import struct
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns

# # Set up argument parser
# parser = argparse.ArgumentParser(description="Train a softmax regression model on the MNIST dataset.")
# parser.add_argument(
#     "-i", "--input", 
#     help="Directory path containing the MNIST dataset files (train-images.idx3-ubyte, train-labels.idx1-ubyte, t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte)", 
#     type=str, 
#     required=True
# )
# parser.add_argument(
#     "-o", "--output", 
#     help="File path for saving the trained model (e.g., softmax_model.pkl)", 
#     type=str, 
#     required=True
# )
# args = parser.parse_args()

# # Validate input file paths
# train_images_path = os.path.join(args.input, 'train-images.idx3-ubyte')
# train_labels_path = os.path.join(args.input, 'train-labels.idx1-ubyte')
# test_images_path = os.path.join(args.input, 't10k-images.idx3-ubyte')
# test_labels_path = os.path.join(args.input, 't10k-labels.idx1-ubyte')

# # Check if the input files exist
# if not os.path.isfile(train_images_path):
#     print(f"Error: Input file path does not exist: {train_images_path}\n")
#     exit(1)

# if not os.path.isfile(train_labels_path):
#     print(f"Error: Input file path does not exist: {train_labels_path}\n")
#     exit(1)

# if not os.path.isfile(test_images_path):
#     print(f"Error: Input file path does not exist: {test_images_path}\n")
#     exit(1)

# if not os.path.isfile(test_labels_path):
#     print(f"Error: Input file path does not exist: {test_labels_path}\n")
#     exit(1)


# Paths to the MNIST dataset files (REMOVE BEFORE SUBMISSION)
train_images_path = 'MNIST\\MNIST\\train-images.idx3-ubyte'  # C
train_labels_path = 'MNIST\\MNIST\\train-labels.idx1-ubyte'  # Update with your path
test_images_path = 'MNIST\\MNIST\\t10k-images.idx3-ubyte'     # Update with your path
test_labels_path = 'MNIST\\MNIST\\t10k-labels.idx1-ubyte'     # Update with your path

# MNIST INFO
# The first 16 bytes of the file contain the header information. Here's how the header is structured:

# Magic Number: 4 bytes

# Indicates the type of data (images or labels).
# Number of Items: 4 bytes

# For image files, this is the number of images.
# For label files, this is the number of labels.
# Number of Rows: 4 bytes (only in image files)

# This indicates the height of each image (28 pixels for MNIST).
# Number of Columns: 4 bytes (only in image files)

# This indicates the width of each image (28 pixels for MNIST).


def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images = struct.unpack('>II', f.read(8)) #reads the first 8 bytes of the file: The first 4 bytes are the magic number - tells data type (which we read but do not use). The next 4 bytes tell us how many images are in the file
        # We don't really use magic so maybe can remove?
        # Read the number of rows and columns
        num_rows, num_cols = struct.unpack('>II', f.read(8))
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
        return images # return images / 255.0  # Do we need to Normalize to [0, 1]...........?

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
# Load the dataset
X_train = read_mnist_images(train_images_path)
y_train = read_mnist_labels(train_labels_path)
X_test = read_mnist_images(test_images_path)
y_test = read_mnist_labels(test_labels_path)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

print("X_train ", X_train)
print("y_train/Labels", y_train)
print("X_test", X_test)
print("y_test/Labels", y_test)

# Function to display a grid of images
def display_images(images, labels, num_images=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        img = images[i].reshape(28, 28)  # Reshape the flat image to 28x28
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')  # Use grayscale colormap
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show first 10 images from the training set
display_images(X_train, y_train, num_images=20)


k = 10  # Number of classes

# T from t
def one_hot_encoding(y_train, k):
    return np.eye(k)[y_train]

y_train = one_hot_encoding(y_train, k)
y_test = one_hot_encoding(y_test, k)

print(y_train) # Print all and see again !!!!

# Softmax mapping function (REWRITE)
W = np.random.randn(k, 784) * 0.01  # small random values

logits = np.dot(W, X_train)
y_pred = np.exp(logits - np.max(logits, axis=0, keepdims=True))/(np.sum(np.exp(np.dot(W, X_train)))) #Matrix shape error!!!!
print(y_pred)

# Loss function for softmax regression (REWRITE)
loss =  -np.sum(y_train * np.log(y_pred))


num_epochs = 10 # Specify number

def training():
    # Double Check
    W = np.random.randn(k, 784) * 0.01  # small random values
    #b = np.zeros((k, 1))  # start with zeros
    print(W.shape)

    for epoch in range(num_epochs):
        Y = forward_pass()
        #loss = loss_function()
        #backward_pass()

        #Update weights

    return W

model = training()

# Save the model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# # Load the model
# with open('trained_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# Calculate accuracy
accuracy = np.mean(np.argmax(y_test, axis=1) == y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(k), yticklabels=np.arange(k))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()