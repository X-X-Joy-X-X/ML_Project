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

# Set up argument parser
parser = argparse.ArgumentParser(description="Test the model trained on MNIST dataset.")
parser.add_argument(
    "-i", "--images", 
    help="Directory path containing the MNIST dataset test images (t10k-images.idx3-ubyte)", 
    type=str, 
    required=True
)
parser.add_argument(
    "-l", "--labels", 
    help="Directory path containing the MNIST dataset test labels (t10k-labels.idx1-ubyte)", 
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
test_labels_path = args.labels

# Check if the input files exist
if not os.path.isfile(test_images_path):
    print(f"Error: Input file path does not exist: {test_images_path}\n")
    exit(1)

if not os.path.isfile(test_labels_path):
    print(f"Error: Input file path does not exist: {test_labels_path}\n")
    exit(1)

if not os.path.isfile(args.model):
    print(f"Error: Model file path does not exist: {args.model}\n")
    exit(1)


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
X_test = read_mnist_images(test_images_path)
y_test = read_mnist_labels(test_labels_path)

def display_images(images, labels, num_images, cols=5):
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(10,5))
    for i in range(num_images):
        img = images[i].reshape(28, 28)  # Reshape image to 28x28
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')  # Grayscale colormap
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show first 20 images from the testing set
display_images(X_test, y_test, num_images=20, cols=5)


k = 10

def one_hot_encoding(y_train, k):
    return np.eye(k)[y_train]

y_test = one_hot_encoding(y_test, k)

def softmax(a):
    exp_a = np.exp(a - np.max(a, axis=0, keepdims=True))
    y_pred = exp_a /(np.sum(exp_a, axis=0, keepdims=True))
    return y_pred.T

# Load the model
with open(args.model, 'rb') as file:
    loaded_model = pickle.load(file)

start_time = time.time()

logits_test = np.dot(loaded_model, X_test.T)  # Shape: (num_samples, k)
Y_preds = softmax(logits_test)  # Shape: (num_samples, k)
y_pred = np.argmax(Y_preds, axis=1)

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

# Write Output to Excel file
output_data = {
    'Image Filename': [f'image_{i}.png' for i in range(len(y_pred))],
    'Label': y_pred
}

df = pd.DataFrame(output_data)
acc_results = pd.DataFrame({'Test Accuracy': [f'{accuracy * 100:.2f}%']})

label_counts = df['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Total Images']

# Save to Excel
output_dir = os.path.dirname(__file__)
output_file_path = os.path.join(output_dir, 'MNIST_Output.xlsx')

# Save to Excel
with pd.ExcelWriter(output_file_path) as writer:
    label_counts.to_excel(writer, sheet_name='Label Counts', index=False)
    acc_results.to_excel(writer, sheet_name='Label Counts', index=False, startcol=5)
    df.to_excel(writer, sheet_name='Image Labels', index=False)
    
print(f"Output is written to Excel file '{output_file_path}'")
