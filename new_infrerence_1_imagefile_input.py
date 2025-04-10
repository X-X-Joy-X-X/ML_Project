import os
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

def softmax(logits):
    """
    Row-wise softmax. logits shape: (N, K)
    Returns probabilities shape: (N, K)
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def load_and_preprocess_image(img_path, target_size=(32,32)):
    """
    Loads and preprocesses a single image:
      - Resizes to target_size
      - Converts to grayscale
      - Enhances contrast & sharpness
      - Normalizes to [0, 1]
      - Flattens into 1D array
    """
    img = Image.open(img_path)

    # Resize if needed
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)

    # Convert to grayscale
    img = img.convert("L")

    # Example image enhancements
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    contrast = ImageEnhance.Contrast(img)
    sharpness = ImageEnhance.Sharpness(img)
    img = contrast.enhance(3)
    img = sharpness.enhance(3)

    arr = np.array(img) / 255.0  # normalize to [0,1]
    return arr.flatten()

def inference_on_all_images_in_folder(
    images_dir,
    model_path,
    target_size=(32,32),
    output_excel="inference_results.xlsx",
    valid_exts=('.png', '.tif', '.jpg')
):
    """
    1) Finds *all* images in 'images_dir' matching valid_exts.
    2) Loads logistic regression model (.pkl).
    3) Infer label for each image.
    4) Saves Excel with columns: [Filename, PredictedLabel]
       plus two extra rows for total label=0 and total label=1.
    """

    # 1) Collect all images
    image_paths = []
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(valid_exts):
            image_paths.append(os.path.join(images_dir, fname))

    if not image_paths:
        print(f"[WARNING] No images found in {images_dir} with extensions {valid_exts}.")
        return

    # 2) Load the trained logistic regression weights
    try:
        with open(model_path, 'rb') as f:
            W = pickle.load(f)  # shape: (k, D)
    except FileNotFoundError:
        print(f"[ERROR] Model not found at: {model_path}")
        return

    filenames = []
    predictions = []

    # 3) Preprocess & classify each image
    for path in image_paths:
        filenames.append(os.path.basename(path))

        x = load_and_preprocess_image(path, target_size)
        x = x.reshape(1, -1)       # shape: (1, D)
        logits = x.dot(W.T)       # shape: (1, k)
        probs = softmax(logits)   # shape: (1, k)
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
    parser = argparse.ArgumentParser(description="Infer on ALL images in one folder (no random selection).")
    parser.add_argument("--images_dir", type=str, help="Path to folder containing images.")
    parser.add_argument("--model_path", type=str, help="Path to trained .pkl model.")
    parser.add_argument("--output_excel", type=str, default="inference_results.xlsx", help="Excel filename to write.")
    parser.add_argument("--target_width", type=int, default=32, help="Target image width (must match training).")
    parser.add_argument("--target_height", type=int, default=32, help="Target image height (must match training).")

    args = parser.parse_args()

    # If arguments not provided, ask user in terminal
    images_dir = args.images_dir
    if not images_dir:
        images_dir = input("Enter path to folder with images: ").strip()

    model_path = args.model_path
    if not model_path:
        model_path = input("Enter path to .pkl model: ").strip()

    output_excel = args.output_excel
    target_size = (args.target_width, args.target_height)

    # Validate folder & .pkl file
    if not images_dir or not os.path.isdir(images_dir):
        print("[ERROR] Invalid images_dir. Please provide a valid folder.")
        return

    if not model_path or not os.path.isfile(model_path):
        print("[ERROR] Invalid model_path. Please provide a valid .pkl file.")
        return

    # Run the inference
    inference_on_all_images_in_folder(
        images_dir=images_dir,
        model_path=model_path,
        target_size=target_size,
        output_excel=output_excel
    )

if __name__ == "__main__":
    main()
