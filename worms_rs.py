import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from PIL import Image

# Just for reproducing results
rd.seed(0)

# Load images from specific directory, compresses them to a smaller size, then converts them to usable values
def load_compress_worms(directory, label, target_size=(101,101)):
	images = []
	labels = []
	filenames = []

	# Search through every file in specified directory
	for filename in os.listdir(directory):
		# Check if file is a png image
		if filename.endswith('.png'):
			img_path = os.path.join(directory, filename)	# Get full path to image
			img = Image.open(img_path)						# Open image

			if img.size != target_size:						# Resize if not already in target size
				img = img.resize(target_size)

			img_array = np.array(img) / 255.0				# Put image pixel values into range of [0, 1)

			flattened_img = img_array.flatten()				# Flatten image into 1-D array

			images.append(flattened_img)
			labels.append(label)
			filenames.append(filename)

	return np.array(images), np.array(labels), filenames

def visualize_image(X, t, target_size):
	idx = rd.randint(0, 5499)
	plt.figure(figsize=(10,3))
	img = X[idx].reshape(target_size)
	plt.imshow(img, cmap='gray')
	plt.title(f't={t} image number {idx}')
	plt.axis('off')
	plt.show()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--none')
	parser.add_argument('-w', '--worms')
	args = parser.parse_args()

	size = (64,64)
	no_worms_dir = args.none
	worms_dir = args.worms

	# THESE DIRECTORIES WORK ON MY PC - MUST BE CHANGED TO ASK USER
	# no_worms_dir = '/home/rbs/Insync/School/Spring 2025/ECE 5370/Project 4/Celegans_ModelGen/0/'
	# worms_dir = '/home/rbs/Insync/School/Spring 2025/ECE 5370/Project 4/Celegans_ModelGen/1/'

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

if __name__ == "__main__":
	main()
