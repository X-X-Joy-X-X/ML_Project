import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import time

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
        return images / 255.0  # Do we need to Normalize to [0, 1]...........?

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

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

def softmax(a):
    exp_a = np.exp(a - np.max(a, axis=0, keepdims=True))
    y_pred = exp_a /(np.sum(exp_a, axis=0, keepdims=True))
    return y_pred.T

# Forward pass
def forward_pass(X, W):
    a = np.dot(W, X.T)  # Shape: (num_samples, k)
    return softmax(a)  # Shape: (num_samples, k)


# Loss function for softmax regression
def softmax_regression_loss(y_pred, y_train):
    return -np.sum(y_train * np.log(y_pred))

# Backward path propogation
def backward_prop(X, Y, T):
    return np.dot(X.T, Y - T)


def training(X, T, k, p, num_epochs):
    N, D = X.shape
    W = np.random.randn(k, D) * 0.01  # for small random values
    print(W.shape)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training Data
        y_pred = forward_pass(X_train, W)
        train_loss = softmax_regression_loss(y_pred, T)
        gradients = backward_prop(X, y_pred, T)
        
        # Update weights
        W = W - p * gradients.T

        # Calculate training accuracy
        train_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(T, axis=1))
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation data
        y_pred_val = forward_pass(X_val, W)
        val_loss = softmax_regression_loss(y_pred_val, y_val)
        val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return W, train_losses, train_accuracies, val_losses, val_accuracies

num_epochs = 600
p = 0.00002 # learning rate

start_time = time.time()

model, train_losses, train_accuracies, val_losses, val_accuracies = training(X_train, y_train, k, p, num_epochs)

end_time = time.time()
training_time = end_time - start_time
print("Training time: ", training_time)

plt.figure(figsize=(12, 5))

# Plotting Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

start_time = time.time()

logits_test = np.dot(loaded_model, X_test.T)  # Shape: (num_samples, k)
Y_test = softmax(logits_test)  # Shape: (num_samples, k)
y_pred = np.argmax(Y_test, axis=1)

end_time = time.time()
testing_time = end_time - start_time
print(f"Testing execution time: {testing_time:.5f} seconds")

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
