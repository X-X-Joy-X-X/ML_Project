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

# ------------------------------------------------------------------------
# DIRECTORIES (adjust these as appropriate on your system)
# ------------------------------------------------------------------------
no_worms_dir = 'C:/DESKTOP_SHIT/Machine Learning/Celegans_ModelGen/0'
worms_dir   = 'C:/DESKTOP_SHIT/Machine Learning/Celegans_ModelGen/1'

# ------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------
batch_size = 128
num_epochs = 500

# Tweak #1: Slightly higher regularization
lambda_reg = 0.0005

# Tweak #2: Higher initial LR
lr_init = 0.001

# Tweak #3: Adjust decay
decay = 0.005

# Number of classes
k = 2

# Hidden layer size
hidden_units = 128   # You can try bigger (e.g. 256) if GPU/CPU allows

# Target resize
size = (32,32)

# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------
def load_compress_worms(directory, label, target_size=(101,101)):
    images = []
    labels = []
    filenames = []

    # Search through every file in specified directory
    idx = 0
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)

            # Resize if not in target size
            if img.size != target_size:
                img = img.resize(target_size, Image.LANCZOS)

            # Example basic enhancements
            contrast = ImageEnhance.Contrast(img)
            sharpness = ImageEnhance.Sharpness(img)
            brightness = ImageEnhance.Brightness(img)

            # Convert to grayscale
            img = img.convert("L")

            # Edge enhance
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

            # Example factor adjustments (could randomize these for data augmentation)
            img = contrast.enhance(4)
            img = sharpness.enhance(3)
            img = brightness.enhance(0.6)

            # Scale to [0, 1]
            img_arr = np.array(img) / 255.0

            # Flatten
            flattened_img = img_arr.flatten()

            images.append(flattened_img)
            labels.append(label)
            filenames.append(filename)

            idx += 1

    return np.array(images), np.array(labels), filenames

def visualize_image(X, t, target_size):
    idx = rd.randint(0, X.shape[0] - 1)
    plt.figure(figsize=(4,3))
    img = X[idx].reshape(target_size)
    plt.imshow(img, cmap='gray')
    plt.title(f'Label={t[idx]}, Image #{idx}')
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

def one_hot_encoding(y_train, k):
    return np.eye(k)[y_train]

# ------------------------------------------------------------------------
# SOFTMAX + ONE HIDDEN LAYER (RELU)
# ------------------------------------------------------------------------
def softmax(logits):
    # logits shape: (batch_size, k)
    exp_a = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

def forward_pass_hidden_layer(X, W1, b1, W2, b2):
    """
    X: (batch_size, D)
    W1: (hidden_units, D)
    b1: (hidden_units,)
    W2: (k, hidden_units)
    b2: (k,)

    Returns:
      - Y_pred: (batch_size, k) after softmax
      - Intermediate values (Z1, A1) for backprop
    """
    # Hidden layer
    Z1 = X.dot(W1.T) + b1  # shape (batch_size, hidden_units)
    A1 = np.maximum(0, Z1) # ReLU

    # Output layer
    Z2 = A1.dot(W2.T) + b2 # shape (batch_size, k)
    Y_pred = softmax(Z2)   # shape (batch_size, k)
    return Y_pred, Z1, A1, Z2

def softmax_regression_loss(Y_pred, T, W1, W2, lambda_reg=0.0):
    """
    Cross-entropy loss with L2 penalty on W1, W2
    Y_pred: (batch_size, k)
    T: (batch_size, k)
    """
    # Cross-entropy
    loss_ce = -np.mean(np.sum(T * np.log(Y_pred + 1e-10), axis=1))

    # Regularization (L2) on W1, W2 only
    reg_loss = 0.5 * lambda_reg * (np.sum(W1*W1) + np.sum(W2*W2))
    return loss_ce + reg_loss

def backward_prop_hidden_layer(X, T, Y_pred, Z1, A1, W1, W2, lambda_reg=0.0):
    """
    Backprop through 1 hidden layer (ReLU)
    Return dW1, db1, dW2, db2
    """
    batch_size = X.shape[0]

    # Output layer gradient
    dZ2 = (Y_pred - T)  # shape (batch_size, k)

    # dW2
    dW2 = dZ2.T.dot(A1) / batch_size  # shape (k, hidden_units)
    dW2 += lambda_reg * W2           # L2 penalty on W2

    # db2
    db2 = np.mean(dZ2, axis=0)  # shape (k,)

    # Backprop to hidden
    dA1 = dZ2.dot(W2)  # shape (batch_size, hidden_units)
    dZ1 = dA1.copy()
    dZ1[Z1 < 0] = 0    # ReLU derivative

    # dW1
    dW1 = dZ1.T.dot(X) / batch_size  # shape (hidden_units, D)
    dW1 += lambda_reg * W1          # L2 penalty on W1

    # db1
    db1 = np.mean(dZ1, axis=0)      # shape (hidden_units,)

    return dW1, db1, dW2, db2

# ------------------------------------------------------------------------
# MINI-BATCH CREATION
# ------------------------------------------------------------------------
def create_batches(X, T, batch_size):
    """
    X: (num_samples, D)
    T: (num_samples, k)
    """
    idx = np.random.permutation(X.shape[0])
    X_shuffle = X[idx]
    T_shuffle = T[idx]

    batches = []
    total_batches = X.shape[0] // batch_size

    for i in range(total_batches):
        batch_X = X_shuffle[i * batch_size:(i + 1) * batch_size]
        batch_T = T_shuffle[i * batch_size:(i + 1) * batch_size]
        batches.append((batch_X, batch_T))

    # Remainder
    if X.shape[0] % batch_size != 0:
        batch_X = X_shuffle[total_batches * batch_size:]
        batch_T = T_shuffle[total_batches * batch_size:]
        batches.append((batch_X, batch_T))

    return batches

def exponential_decay(epoch, initial_lr=lr_init, decay_rate=decay):
    """
    Exponential decay of learning rate.
    """
    return initial_lr * np.exp(-decay_rate * epoch)

# ------------------------------------------------------------------------
# TRAINING FUNCTION (with hidden layer)
# ------------------------------------------------------------------------
def training_with_hidden_layer(X_train, X_val, T_train, T_val,
                              input_dim, hidden_units, k,
                              lambda_reg=0.0, num_epochs=100, batch_size=128):
    """
    We'll keep the same structure but now we have W1,b1, W2,b2.
    """
    # Initialize parameters
    # W1: (hidden_units, input_dim)
    # b1: (hidden_units,)
    # W2: (k, hidden_units)
    # b2: (k,)
    W1 = 0.01 * np.random.randn(hidden_units, input_dim)
    b1 = np.zeros(hidden_units)
    W2 = 0.01 * np.random.randn(k, hidden_units)
    b2 = np.zeros(k)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Decayed LR
        lr = exponential_decay(epoch)

        epoch_train_loss = 0.0
        epoch_train_acc  = 0.0
        num_batches = 0

        # Create mini-batches
        batches = create_batches(X_train, T_train, batch_size)

        for (X_b, T_b) in batches:
            # Forward pass
            Y_b_pred, Z1_b, A1_b, Z2_b = forward_pass_hidden_layer(X_b, W1, b1, W2, b2)

            # Compute batch loss
            batch_loss = softmax_regression_loss(Y_b_pred, T_b, W1, W2, lambda_reg)
            epoch_train_loss += batch_loss

            # Accuracy (batch)
            batch_acc = np.mean(np.argmax(Y_b_pred, axis=1) == np.argmax(T_b, axis=1))
            epoch_train_acc += batch_acc

            # Backprop
            dW1, db1, dW2, db2 = backward_prop_hidden_layer(
                X_b, T_b, Y_b_pred, Z1_b, A1_b, W1, W2, lambda_reg
            )

            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

            num_batches += 1

        # Averages for epoch
        epoch_train_loss /= num_batches
        epoch_train_acc  /= num_batches
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation
        Y_val_pred, _, _, _ = forward_pass_hidden_layer(X_val, W1, b1, W2, b2)
        val_loss = softmax_regression_loss(Y_val_pred, T_val, W1, W2, lambda_reg)
        val_accuracy = np.mean(np.argmax(Y_val_pred, axis=1) == np.argmax(T_val, axis=1))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | LR={lr:.6f} | "
                  f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    # Return the final parameters + training history
    return (W1, b1, W2, b2), train_losses, train_accuracies, val_losses, val_accuracies

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    print("Reading no worms...")
    X_no_worms, t_no_worms, no_worms_names = load_compress_worms(no_worms_dir, 0, target_size=size)
    print(f"Images read (no worms): {X_no_worms.shape[0]}")
    print(f"Image size (flattened): {X_no_worms.shape[1]}")

    print("Reading worms...")
    X_worms, t_worms, worms_names = load_compress_worms(worms_dir, 1, target_size=size)
    print(f"Images read (worms): {X_worms.shape[0]}")
    print(f"Image size (flattened): {X_worms.shape[1]}")

    # Quick visualization
    visualize_image(X_no_worms, t_no_worms, size)
    visualize_image(X_worms, t_worms, size)

    # Combine data
    # X_no_worms: (num_no_worms, D)
    # X_worms:    (num_worms, D)
    # We'll stack them vertically and create X of shape (N, D)
    X = np.vstack((X_no_worms, X_worms))
    t_labels = np.concatenate((t_no_worms, t_worms))

    print("Combined X shape:", X.shape)      # (total_samples, D)
    print("Combined t shape:", t_labels.shape)  # (total_samples,)

    # Split into train/test
    X_train, X_test, t_train_raw, t_test_raw = train_test_split(
        X, t_labels, test_size=0.2, random_state=42
    )

    visualize_split(t_train_raw, t_test_raw)

    # Further split train into train/val
    X_train, X_val, t_train_raw, t_val_raw = train_test_split(
        X_train, t_train_raw, test_size=0.25, random_state=42
    )

    # 1.2 More Robust Normalization
    # Compute mean/std from X_train only, then apply to X_train, X_val, X_test
    mean_train = np.mean(X_train, axis=0, keepdims=True)
    std_train  = np.std(X_train, axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean_train) / std_train
    X_val   = (X_val   - mean_train) / std_train
    X_test  = (X_test  - mean_train) / std_train

    # One-hot encode
    T_train = one_hot_encoding(t_train_raw, k)  # shape (num_train, k)
    T_val   = one_hot_encoding(t_val_raw, k)    # shape (num_val, k)
    T_test  = one_hot_encoding(t_test_raw, k)   # shape (num_test, k)

    # Train with hidden layer
    input_dim = X_train.shape[1]  # D
    start_time = time.time()

    (W1, b1, W2, b2), train_losses, train_accuracies, val_losses, val_accuracies = training_with_hidden_layer(
        X_train, X_val, T_train, T_val,
        input_dim=input_dim,
        hidden_units=hidden_units,
        k=k,
        lambda_reg=lambda_reg,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs (Hidden Layer Model)')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies,   label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs (Hidden Layer Model)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save model
    with open('trained_worms_hidden.pkl', 'wb') as file:
        pickle.dump((W1, b1, W2, b2, mean_train, std_train), file)

    # ------------------------------
    # TEST PHASE
    # ------------------------------
    start_time = time.time()

    # Forward pass on test
    Y_test_pred, _, _, _ = forward_pass_hidden_layer(X_test, W1, b1, W2, b2)
    y_pred = np.argmax(Y_test_pred, axis=1)       # shape (num_test,)

    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing execution time: {testing_time:.5f} seconds")

    # Calculate accuracy
    accuracy = np.mean(np.argmax(T_test, axis=1) == y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Confusion matrix
    cm = confusion_matrix(np.argmax(T_test, axis=1), y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Worms', 'Worms'],
                yticklabels=['No Worms', 'Worms'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Hidden Layer Model')
    plt.show()

if __name__ == '__main__':
    main()
