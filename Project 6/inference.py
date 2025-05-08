import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_images(directory, target_size=(32, 32)):
    images = []
    filenames = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            path = os.path.join(directory, filename)
            img = Image.open(path)
            img = img.resize(target_size, Image.LANCZOS)
            img = img.convert("L")  # Grayscale
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

            contrast = ImageEnhance.Contrast(img)
            sharpness = ImageEnhance.Sharpness(img)
            brightness = ImageEnhance.Brightness(img)
            
            img = contrast.enhance(4)
            img = sharpness.enhance(3)
            img = brightness.enhance(0.6)
    
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_arr = np.expand_dims(img_arr, axis=-1)  # Add channel dimension (32, 32, 1)

            images.append(img_arr)
            filenames.append(filename)
            # === Remove this code block before submitting ===
            if filename.endswith('NW.png'):
               label = 0
            elif filename.endswith('W.png'):
               label = 1
            else:
                label = -1
            labels.append(label)
            # ========================================

    return np.array(images), filenames, labels

def run_inference(img_dir, model_path, output_excel="CNN_Inference.xlsx"):
    """
    1. Get images from image directory
    2. Load CNN model
    3. Infer label for each selected image
    4. Save Excel with columns: [Filename, PredictedLabel]
       plus rows for total label=0 and total label=1
    """
    # Load model
    model = load_model(model_path)

    # Load and preprocess images
    X_test, filenames, T_test = load_and_preprocess_images(img_dir)
    if len(X_test) == 0:
        print("No .png images found in the directory.")
        return

    # Run inference
    pred = model.predict(X_test, verbose=0)
    Y_test = np.argmax(pred, axis=1)  # Get predicted labels for all images
    
    # Calculate accuracy ---------- Maybe Delete before submitting
    if (len(Y_test) > 0 and T_test[0]!=-1):
        Y_test_categorical = to_categorical(T_test, num_classes=2)
        results = model.evaluate(X_test, Y_test_categorical, verbose=0)
        if isinstance(results, list):
            test_loss, test_acc = results[0], results[1]  # If evaluate returns multiple metrics
        else:
            test_loss, test_acc = results, None  # Handle case where only loss is returned
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
    else:
        test_acc = None
        print("Labels unavailable to calculate accuracy.")

    # Build DataFrame
    df = pd.DataFrame({
        "Filename": filenames,
        "PredictedLabel": Y_test
    })

    # Calculate totals
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

    # Save to Excel
    df_final.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

def main():
    parser = argparse.ArgumentParser(description="Run CNN inference on images.")
    parser.add_argument('-i', '--images', help="Directory containing .png images")
    parser.add_argument('-p', '--pickle', help="Path to the trained model (.h5 file)")
    args = parser.parse_args()

    img_dir = args.images
    model_path = args.pickle

    # If arguments are not provided, prompt user
    if not img_dir:
        img_dir = input("Enter images directory: ")
    
    if not model_path:
        model_path = input("Enter model path (.h5): ")

    # Validate inputs
    if not os.path.isdir(img_dir):
        print("Error: Invalid image directory.")
        return
    if not os.path.isfile(model_path):
        print("Error: Invalid model path.")
        return

    run_inference(img_dir, model_path)

if __name__ == "__main__":
    main()
