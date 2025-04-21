import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import struct
import pandas as pd
from PIL import Image

# Set up argument parser
parser = argparse.ArgumentParser(description="Test the model trained on MNIST dataset.")
parser.add_argument(
    "-i", "--images", 
    help="Directory path containing the MNIST dataset test images (t10k-images.idx3-ubyte)", 
    type=str, 
    required=True
)
parser.add_argument(
    "-m", "--model", 
    help="Directory path containing the trained model", 
    type=str, 
    required=True
)

args = parser.parse_args()

test_images_path = args.images

# Check if the model file exists
if not os.path.isfile(args.model):
    print(f"Error: Model file path does not exist: {args.model}\n")
    exit(1)

def load_and_preprocess_image(img_path):
    """Load and preprocess an image from path."""
    img = Image.open(img_path)
    arr = np.array(img) / 255.0
    arr = arr.flatten()
    return arr

def read_mnist_images(file_path):
    """Read MNIST test images and sort them numerically."""
    image_paths = []
    for fname in os.listdir(file_path):
        if fname.lower().endswith('.tif'):
            image_paths.append(os.path.join(file_path, fname))
    
    # Sort by numeric part of filename to ensure consistency
    image_paths.sort(key=lambda x: int(os.path.basename(x).replace('img', '').replace('.tif', '')))
    print(f"Found {len(image_paths)} image files")
    return image_paths

def read_mnist_labels(file_path):
    """Read MNIST test labels from a text file."""
    labels_found = False
    for fname in os.listdir(file_path):
        if fname.lower().endswith('.txt'):
            path = os.path.join(file_path, fname)
            labels = np.loadtxt(path, dtype=int)
            labels_found = True
            print(f"Loaded {len(labels)} labels from {fname}")
            break
    
    if not labels_found:
        print("No label file found with .txt extension")
        exit(1)
        
    return labels

def display_images(images, labels, num_images, cols=5):
    """Display a grid of images with their labels."""
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(10,5))
    for i in range(min(num_images, len(images))):
        img = images[i].reshape(28, 28)  # Reshape image to 28x28
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')  # Grayscale colormap
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def one_hot_encoding(y, k):
    """Convert labels to one-hot encoding."""
    return np.eye(k)[y]

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Load the dataset and model
image_paths = read_mnist_images(test_images_path)
y_test = read_mnist_labels(test_images_path)

# Check that number of images matches number of labels
if len(image_paths) != len(y_test):
    print(f"Warning: Number of images ({len(image_paths)}) doesn't match number of labels ({len(y_test)})")
    # Truncate to the smaller of the two
    min_count = min(len(image_paths), len(y_test))
    image_paths = image_paths[:min_count]
    y_test = y_test[:min_count]

# Process images
filenames = []
X_test = []

for path in image_paths:
    filename = os.path.basename(path)
    filenames.append(filename)
    x = load_and_preprocess_image(path)
    X_test.append(x)

X_test = np.array(X_test)

# Show first 20 images from the testing set
display_images(X_test, y_test, num_images=20, cols=5)

# Define number of classes
k = 10

# One-hot encode the labels
y_test_one_hot = one_hot_encoding(y_test, k)

# Load the model
with open(args.model, 'rb') as file:
    loaded_model = pickle.load(file)

# Print shapes for debugging
print(f"Model shape: {loaded_model.shape}")
print(f"X_test shape: {X_test.shape}")

start_time = time.time()

# Adjust the matrix multiplication based on the model shape
if loaded_model.shape[1] == X_test.shape[1]:
    # Model is (10, 784) and X_test is (num_samples, 784)
    logits_test = np.dot(X_test, loaded_model.T)
elif loaded_model.shape[0] == X_test.shape[1]:
    # Model is (784, 10) and X_test is (num_samples, 784)
    logits_test = np.dot(X_test, loaded_model)
else:
    # Try transposing the model if dimensions don't match
    print(f"Warning: Dimensions don't align. Attempting to transpose.")
    logits_test = np.dot(X_test, loaded_model.T)

# Apply softmax to get probabilities
Y_preds = softmax(logits_test)
y_pred = np.argmax(Y_preds, axis=1)

end_time = time.time()
testing_time = end_time - start_time
print(f"Testing execution time: {testing_time:.5f} seconds")

# Calculate accuracy
y_test_argmax = np.argmax(y_test_one_hot, axis=1)
accuracy = np.mean(y_test_argmax == y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Print more details for debugging
print(f"Ground truth distribution: {np.bincount(y_test_argmax, minlength=10)}")
print(f"Prediction distribution: {np.bincount(y_pred, minlength=10)}")

# Confusion matrix
cm = confusion_matrix(y_test_argmax, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(k), yticklabels=np.arange(k))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

output_data = {
    'Image Filename': filenames,
    'Label': y_pred
}

df = pd.DataFrame(output_data)
acc_results = pd.DataFrame({'Test Accuracy': [f'{accuracy * 100:.2f}%']})

label_counts = df['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Total Images']

# Save to Excel
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(output_dir, 'MNIST_Output.xlsx')

# Save to Excel - simplified format with just filename and predicted label
with pd.ExcelWriter(output_file_path) as writer:
    label_counts.to_excel(writer, sheet_name='Label Counts', index=False)
    acc_results.to_excel(writer, sheet_name='Label Counts', index=False, startcol=5)
    df.to_excel(writer, sheet_name='Image Labels', index=False)
    
print(f"Output is written to Excel file '{output_file_path}'")
