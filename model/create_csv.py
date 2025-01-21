import os
import pandas as pd
from PIL import Image
import numpy as np

# Define the dataset directory
DATASET_DIR = 'dataset'

# Define the emotion labels (based on folder names)
LABELS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def process_images(data_dir, output_csv, chunk_size=500):
    """
    Processes images in a given directory and saves them to a CSV file.
    The function processes data in chunks for better performance on large datasets.

    Parameters:
        data_dir (str): Path to the dataset directory.
        output_csv (str): Path to save the output CSV.
        chunk_size (int): Number of records to process in each batch.
    """
    data = []
    chunk_count = 0

    # Iterate over each label folder
    for label_name, label_index in LABELS.items():
        folder_path = os.path.join(data_dir, label_name)

        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder '{label_name}' not found in '{data_dir}', skipping.")
            continue

        # Process each image in the folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                # Open the image and convert to grayscale
                img = Image.open(img_path).convert('L')  # 'L' mode for grayscale
                img = img.resize((48, 48))  # Resize to 48x48

                # Flatten the image into a single array of pixel values
                pixels = np.array(img).flatten()

                # Convert the pixel array to a space-separated string
                pixels_str = ' '.join(map(str, pixels))

                # Append the data (label, pixels) to the list
                data.append([label_index, pixels_str])

                # Save in chunks to avoid memory issues
                if len(data) >= chunk_size:
                    chunk_df = pd.DataFrame(data, columns=["emotion", "pixels"])
                    chunk_df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
                    data = []  # Clear the buffer
                    chunk_count += 1
                    print(f"Processed chunk {chunk_count} for '{label_name}'.")

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    # Save any remaining data
    if data:
        chunk_df = pd.DataFrame(data, columns=["emotion", "pixels"])
        chunk_df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
        print(f"Final chunk processed for '{data_dir}'.")

    print(f"CSV file created: {output_csv}")

# Process train and test datasets
if __name__ == "__main__":
    # Clear existing CSV files to avoid appending to old data
    if os.path.exists('train.csv'):
        os.remove('train.csv')
    if os.path.exists('test.csv'):
        os.remove('test.csv')

    process_images(os.path.join(DATASET_DIR, 'train'), 'train.csv')
    process_images(os.path.join(DATASET_DIR, 'test'), 'test.csv')
