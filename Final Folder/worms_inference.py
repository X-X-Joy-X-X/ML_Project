import os
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import tkinter as tk
from tkinter import filedialog, messagebox



###############################################
#    HELPER FUNCTIONS
###############################################

def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def load_and_preprocess_image(img_path, target_size=(32,32)):
    img = Image.open(img_path)
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)

    img = img.convert("L")
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    contrast = ImageEnhance.Contrast(img)
    sharpness = ImageEnhance.Sharpness(img)
    img = contrast.enhance(3)
    img = sharpness.enhance(3)

    arr = np.array(img) / 255.0
    return arr.flatten()

def pick_random_images(folder, num_to_pick, valid_exts=('.png', '.tif', '.jpg')):
    """
    Gathers all image files from 'folder' with valid extensions
    then returns a random sample of size 'num_to_pick' (or fewer if not enough).
    """
    all_imgs = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(valid_exts):
            all_imgs.append(os.path.join(folder, fname))

    if len(all_imgs) == 0:
        print(f"No images found in {folder} with {valid_exts}.")
        return []

    if len(all_imgs) <= num_to_pick:
        return all_imgs

    return random.sample(all_imgs, num_to_pick)

def random_test_two_folders(
    worms_dir,
    no_worms_dir,
    model_path,
    num_from_worms=5,
    num_from_no_worms=5,
    target_size=(32,32),
    output_excel="random_inference_two_folders.xlsx"
):
    """
    1. Pick random images from 'worms_dir' and 'no_worms_dir'
    2. Load logistic regression model
    3. Infer label for each selected image
    4. Save Excel with columns: [Filename, SourceFolder, PredictedLabel]
       plus rows for total label=0 and total label=1
    """
    worms_imgs = pick_random_images(worms_dir, num_from_worms)
    worms_folders = ["worms"] * len(worms_imgs)

    no_worms_imgs = pick_random_images(no_worms_dir, num_from_no_worms)
    no_worms_folders = ["no_worms"] * len(no_worms_imgs)

    selected_image_paths = worms_imgs + no_worms_imgs
    selected_folders = worms_folders + no_worms_folders

    if not selected_image_paths:
        messagebox.showwarning("No Images", "No images were selected from the folders.")
        return

    try:
        with open(model_path, 'rb') as f:
            W = pickle.load(f)
    except FileNotFoundError:
        messagebox.showerror("Model Not Found", f"Could not open model file:\n{model_path}")
        return

    filenames = []
    folder_sources = []
    predictions = []

    for path, folder_label in zip(selected_image_paths, selected_folders):
        filenames.append(os.path.basename(path))
        folder_sources.append(folder_label)

        x = load_and_preprocess_image(path, target_size)
        x = x.reshape(1, -1)
        logits = x.dot(W.T)  # shape: (1, k)
        probs = softmax(logits)
        pred_label = np.argmax(probs, axis=1)[0]
        predictions.append(pred_label)

    # Build DataFrame
    df = pd.DataFrame({
        "Filename": filenames,
        "SourceFolder": folder_sources,
        "PredictedLabel": predictions
    })

    # Totals
    label_counts = df["PredictedLabel"].value_counts().to_dict()
    total_rows = []
    for label_val in sorted(label_counts.keys()):
        row_dict = {
            "Filename": f"Total images for label {label_val}",
            "SourceFolder": "",
            "PredictedLabel": label_counts[label_val]
        }
        total_rows.append(row_dict)

    df_totals = pd.DataFrame(total_rows)
    df_final = pd.concat([df, df_totals], ignore_index=True)

    df_final.to_excel(output_excel, index=False)
    print("Done")

def run_inference(w_dir, nw_dir, pkl):
    # NOTE: You may want to choose how many images from each folder:
    num_from_worms = 5
    num_from_no_worms = 5

    # Now call the random inference function
    random_test_two_folders(
        worms_dir=w_dir,
        no_worms_dir=nw_dir,
        model_path=pkl,
        num_from_worms=num_from_worms,
        num_from_no_worms=num_from_no_worms,
        target_size=(32,32),  # must match your training image size
        output_excel="worm_random_inference_results.xlsx"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--none')
    parser.add_argument('-w', '--worms')
    parser.add_argument('-p', '--pickle')
    args = parser.parse_args()

    w_dir = args.worms
    nw_dir = args.none
    pkl = args.pickle

    print(w_dir)
    print(nw_dir)
    print(pkl)

    if not w_dir:
        w_dir = input("Enter worms directory: ")

    if not nw_dir:
        nw_dir = input("Enter no worms directory: ")
    
    if not pkl:    
        pkl = input("Enter model directory (.pkl): ")

    if not w_dir or not nw_dir or not pkl:
        print("Missing Info", "Please select all three paths before running.")
        return

    run_inference(w_dir, nw_dir, pkl)

if __name__ == "__main__":
    main()
