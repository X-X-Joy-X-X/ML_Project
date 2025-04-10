import os
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

def forward_pass_hidden_layer(X, W1, b1, W2, b2):

    # Hidden layer
    Z1 = X.dot(W1.T) + b1  # (N, hidden_units)
    A1 = np.maximum(0, Z1) # ReLU

    # Output layer
    Z2 = A1.dot(W2.T) + b2 # (N, k)

    # Row-wise softmax
    shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def load_and_preprocess_image(img_path, target_size=(32, 32), mean_train=None, std_train=None):
 
    img = Image.open(img_path)

    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)

    img = img.convert("L")

    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    contrast = ImageEnhance.Contrast(img)
    sharpness = ImageEnhance.Sharpness(img)
    brightness = ImageEnhance.Brightness(img)

    img = contrast.enhance(4)
    img = sharpness.enhance(3)
    img = brightness.enhance(0.6)

    arr = np.array(img) / 255.0

    if (mean_train is not None) and (std_train is not None):
       
        arr = arr.flatten()
        arr = (arr - mean_train.ravel()) / (std_train.ravel() + 1e-8)
    else:
        arr = arr.flatten()

    return arr

def inference_on_all_images_in_folder(
    images_dir,
    model_path,
    target_size=(32, 32),
    output_excel="inference_results.xlsx",
    valid_exts=('.png', '.tif', '.jpg', '.jpeg')
):

    # 1) Gather all images
    image_paths = []
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(valid_exts):
            image_paths.append(os.path.join(images_dir, fname))

    if not image_paths:
        print(f"[WARNING] No images found in {images_dir} with extensions {valid_exts}.")
        return

    # 2) Load the trained model
    #    The training code pickles: (W1, b1, W2, b2, mean_train, std_train)
    try:
        with open(model_path, 'rb') as f:
            W1, b1, W2, b2, mean_train, std_train = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Model not found at: {model_path}")
        return
    except ValueError:
        print("[ERROR] The .pkl file does not contain the expected 6-element tuple.")
        return

    # Collect filenames and predictions
    filenames = []
    predictions = []

    # 3) Preprocess & classify each image
    for path in image_paths:
        filenames.append(os.path.basename(path))

        # Prepare the image exactly as in training
        x = load_and_preprocess_image(
            path,
            target_size=target_size,
            mean_train=mean_train,
            std_train=std_train
        )

        # x shape: (D,)
        x = x.reshape(1, -1)  # reshape to (1, D)

        # Forward pass with hidden layer
        probs = forward_pass_hidden_layer(x, W1, b1, W2, b2)
        pred_label = np.argmax(probs, axis=1)[0]  # 0 or 1
        predictions.append(pred_label)

    # 4) Build a DataFrame with 2 columns
    df = pd.DataFrame({
        "Filename": filenames,
        "PredictedLabel": predictions
    })

    # Totals for each label
    label_counts = df["PredictedLabel"].value_counts().to_dict()
    total_rows = []
    for label_val in sorted(label_counts.keys()):
        row_dict = {
            "Filename": f"Total images for label {label_val}",
            "PredictedLabel": label_counts[label_val]
        }
        total_rows.append(row_dict)

    df_totals = pd.DataFrame(total_rows)
    df_final = pd.concat([df, df_totals], ignore_index=True)

    df_final.to_excel(output_excel, index=False)
    print(f"[INFO] Inference complete. Results saved to '{output_excel}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Infer on ALL images in one folder (no random selection)."
    )
    parser.add_argument("--images_dir", type=str, help="Path to folder containing images.")
    parser.add_argument("--model_path", type=str, help="Path to trained .pkl model.")
    parser.add_argument("--output_excel", type=str, default="inference_results.xlsx", help="Excel filename to write.")
    parser.add_argument("--target_width", type=int, default=32, help="Target image width (must match training).")
    parser.add_argument("--target_height", type=int, default=32, help="Target image height (must match training).")

    args = parser.parse_args()

    images_dir = args.images_dir
    if not images_dir:
        images_dir = input("Enter path to folder with images: ").strip()

    model_path = args.model_path
    if not model_path:
        model_path = input("Enter path to .pkl model: ").strip()

    output_excel = args.output_excel
    target_size = (args.target_width, args.target_height)

    if not images_dir or not os.path.isdir(images_dir):
        print("[ERROR] Invalid images_dir. Please provide a valid folder.")
        return

    if not model_path or not os.path.isfile(model_path):
        print("[ERROR] Invalid model_path. Please provide a valid .pkl file.")
        return

    inference_on_all_images_in_folder(
        images_dir=images_dir,
        model_path=model_path,
        target_size=target_size,
        output_excel=output_excel
    )

if __name__ == "__main__":
    main()
