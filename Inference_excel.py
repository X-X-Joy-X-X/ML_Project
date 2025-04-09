import os
import random
import pickle
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

def load_and_preprocess_image(img_path, target_size=(25,25)):
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
    target_size=(25,25),
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
    messagebox.showinfo("Done", f"Results saved to: {output_excel}")

###############################################
#    TKINTER UI
###############################################

class RandomInferenceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Inference UI")

        self.worms_dir = tk.StringVar()
        self.no_worms_dir = tk.StringVar()
        self.pkl_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Worms Directory
        tk.Label(self.root, text="Worms Folder:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.worms_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_worms).grid(row=0, column=2, padx=5, pady=5)

        # No Worms Directory
        tk.Label(self.root, text="No-Worms Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.no_worms_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_no_worms).grid(row=1, column=2, padx=5, pady=5)

        # PKL File
        tk.Label(self.root, text="Model PKL File:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.pkl_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_pkl).grid(row=2, column=2, padx=5, pady=5)

        # Run Inference Button
        tk.Button(self.root, text="Run Inference", command=self.run_inference).grid(
            row=3, column=0, columnspan=3, padx=5, pady=10
        )

    def browse_worms(self):
        folder = filedialog.askdirectory(title="Select Worms Folder")
        if folder:
            self.worms_dir.set(folder)

    def browse_no_worms(self):
        folder = filedialog.askdirectory(title="Select No-Worms Folder")
        if folder:
            self.no_worms_dir.set(folder)

    def browse_pkl(self):
        file = filedialog.askopenfilename(
            title="Select PKL Model File",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        if file:
            self.pkl_path.set(file)

    def run_inference(self):
        # Validate
        w_dir = self.worms_dir.get().strip()
        nw_dir = self.no_worms_dir.get().strip()
        pkl = self.pkl_path.get().strip()

        if not w_dir or not nw_dir or not pkl:
            messagebox.showwarning("Missing Info", "Please select all three paths before running.")
            return

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
            target_size=(60,60),  # must match your training image size
            output_excel="random_inference_results.xlsx"
        )

def main():
    root = tk.Tk()
    app = RandomInferenceUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
