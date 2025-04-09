import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import pickle
import random as rd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

no_worms_dir = '/home/rbs/Insync/School/Spring 2025/ECE 5370/Project 4/Celegans_ModelGen/0/'
worms_dir = '/home/rbs/Insync/School/Spring 2025/ECE 5370/Project 4/Celegans_ModelGen/1/'

batch_size = 128
num_epochs = 1000
lambda_reg = 0.001
lr_init = 0.0002 # learning rate
decay = 0.02
k = 2
size = (25,25)

def load_compress_worms(directory, label, target_size=(101,101)):
	images = []
	labels = []
	filenames = []

	means = []
	stds = []

	# Search through every file in specified directory
	idx = 0
	for filename in os.listdir(directory):
		# Check if file is a png image
		if filename.endswith('.png') and idx != 5450:
			img_path = os.path.join(directory, filename)	# Get full path to image
			img = Image.open(img_path)						# Open image

			if img.size != target_size:						# Resize if not already in target size
				img = img.resize(target_size, Image.LANCZOS)

			contrast = ImageEnhance.Contrast(img)
			sharpness = ImageEnhance.Sharpness(img)
			brightness = ImageEnhance.Brightness(img)

			img = img.convert("L")
			img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

			img = contrast.enhance(3)
			img = sharpness.enhance(3)

			img_arr = np.array(img)	/ 255		# Put image pixel values into range of [0, 1)

			flattened_img = img_arr.flatten()

			images.append(flattened_img)
			labels.append(label)
			filenames.append(filename)

			idx += 1

	return np.array(images), np.array(labels), filenames

def visualize_image(X, t, target_size):
	idx = rd.randint(0, 5400)
	plt.figure(figsize=(10,3))
	img = X[idx].reshape(target_size)
	plt.imshow(img, cmap='gray')
	plt.title(f't={t} image number {idx}')
	plt.axis('off')
	plt.show()


def visualize_split(t_train, t_test):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.pie(np.unique(t_train, return_counts=True)[1], 
            labels=['No Worms', 'Worms'], autopct='%1.1f%%')
    plt.title('Training Set Distribution')
    
    plt.subplot(1, 2, 2)
    plt.pie(np.unique(t_test, return_counts=True)[1], 
            labels=['No Worms', 'Worms'], autopct='%1.1f%%')
    plt.title('Test Set Distribution')
    
    plt.tight_layout()
    plt.show()


# After combining datasets (X and t)
# X shape= (4096, m), t shape: (m,)
def train_test_split(X, t, test_size=0.2):
    # Set random seed for reproducibility
    #np.random.seed(random_state)
    
    # Create shuffled indices
    m = X.shape[1]
    indices = np.random.permutation(m)
    
    # Calculate split index
    split = int(m * (1 - test_size))
    
    # Split indices
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    # Split data
    X_train = X[:, train_idx]
    t_train = t[train_idx]
    X_test = X[:, test_idx]
    t_test = t[test_idx]
    
    return X_train, X_test, t_train, t_test

def one_hot_encoding(y_train, k):
    return np.eye(k)[y_train]

def softmax(a):
    exp_a = np.exp(a - np.max(a, axis=0, keepdims=True))
    y_pred = exp_a /(np.sum(exp_a, axis=0, keepdims=True))
    return y_pred.T

# Forward pass
def forward_pass(X, W):
    a = np.dot(W, X.T)  # Shape: (num_samples, k)
    return softmax(a)  # Shape: (num_samples, k)

# Loss function for softmax regression
def softmax_regression_loss(y_pred, y_train, W, lambda_reg=0.01):
    loss_func = -np.mean(y_train * np.log(y_pred + 1e-10))
    reg_loss = 0.5 * lambda_reg * np.sum(W * W)
    return loss_func + reg_loss

# Backward path propogation
def backward_prop(X, Y, T, W, lambda_reg=0.01):
    grad = np.dot((Y-T).T, X)
    grad += 2 * lambda_reg * W
    return grad

def create_batches(X, T, batch_size):
    idx = np.random.permutation(X.shape[0])
    X_shuffle = X[idx]
    T_shuffle = T[idx]

    batches = []
    total_batches = X.shape[0] // batch_size

    for i in range(total_batches):
        batch_X = X_shuffle[i * batch_size:(i + 1) * batch_size]
        batch_T = T_shuffle[i * batch_size:(i + 1) * batch_size]
        batches.append((batch_X, batch_T))

    if X.shape[0] % batch_size != 0:
        batch_X = X_shuffle[total_batches * batch_size:]
        batch_T = T_shuffle[total_batches * batch_size:]
        batches.append((batch_X, batch_T))

    return batches

def exponential_decay(epoch, initial_lr=lr_init, decay_rate=decay):  # Slower decay
    return initial_lr * np.exp(-decay_rate * epoch)

def training(X, X_val, t, t_val, k, lambda_reg=0.01, num_epochs=100, batch_size=128):
    D = X.shape[0]
    W = np.random.randn(k, D) * 0.01  # for small random values
    print(W.shape)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        lr = exponential_decay(epoch)

        epoch_train_loss = 0
        epoch_train_acc = 0
        num_batches = 0

        batches = create_batches(X.T, t, batch_size)

        for batch in batches:
            X_b, T_b = batch

            # Training Data
            y_pred = forward_pass(X_b, W)
            batch_loss = softmax_regression_loss(y_pred, T_b, W, lambda_reg)
            epoch_train_loss += batch_loss

            batch_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(T_b, axis=1))
            epoch_train_acc += batch_acc

            gradients = backward_prop(X_b, y_pred, T_b, W, lambda_reg)
        
            # Update weights
            W = W - lr * gradients

            num_batches += 1

        epoch_train_loss /= num_batches
        epoch_train_acc /= num_batches

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation data
        y_pred_val = forward_pass(X_val.T, W)
        val_loss = softmax_regression_loss(y_pred_val, t_val, W, lambda_reg)
        val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == np.argmax(t_val, axis=1))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, LR {lr}, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return W, train_losses, train_accuracies, val_losses, val_accuracies

def main():
	print("Reading no worms...")
	X_no_worms, t_no_worms, no_worms_names = load_compress_worms(no_worms_dir, 0, target_size=size)
	print(f"Images read: {X_no_worms.shape[0]}")
	print(f"Image sizes: {X_no_worms.shape[1]}")

	print("Reading worms...")
	X_worms, t_worms, worms_names = load_compress_worms(worms_dir, 1, target_size=size)
	print(f"Images read: {X_worms.shape[0]}")
	print(f"Image sizes: {X_worms.shape[1]}")

	visualize_image(X_no_worms, t_no_worms, size)
	visualize_image(X_worms, t_worms, size)
	
	X = np.hstack((X_no_worms.T, X_worms.T))  # (4096, m)
	t = np.concatenate((t_no_worms, t_worms))

	print(X.shape)
    
    # Split into train/test
	X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)

	print(X_train.shape)

	t_train = one_hot_encoding(t_train, k)
	t_test = one_hot_encoding(t_test, k)
    
    # Split train into train/val
	X_train, X_val, t_train, t_val = train_test_split(X_train, t_train, test_size=0.25)

	print(X_train.shape)

	start_time = time.time()

	model, train_losses, train_accuracies, val_losses, val_accuracies = training(X_train, X_val, t_train, t_val, k, lambda_reg, num_epochs, batch_size)

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
	with open('trained_worms.pkl', 'wb') as file:
	    pickle.dump(model, file)

	# Load the model
	with open('trained_worms.pkl', 'rb') as file:
	    loaded_model = pickle.load(file)

	start_time = time.time()

	logits_test = np.dot(loaded_model, X_test)  # Shape: (num_samples, k)
	Y_test = softmax(logits_test)  # Shape: (num_samples, k)
	y_pred = np.argmax(Y_test, axis=1)

	end_time = time.time()
	testing_time = end_time - start_time
	print(f"Testing execution time: {testing_time:.5f} seconds")

	# Calculate accuracy
	accuracy = np.mean(np.argmax(t_test, axis=1) == y_pred)
	print(f'Test Accuracy: {accuracy * 100:.2f}%')

	# Confusion matrix
	cm = confusion_matrix(np.argmax(t_test, axis=1), y_pred)
	plt.figure(figsize=(10, 7))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(k), yticklabels=np.arange(k))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('Confusion Matrix')
	plt.show()

if __name__ == '__main__':
	main()