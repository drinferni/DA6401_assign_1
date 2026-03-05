import wandb
import idx2numpy
import numpy as np
import os

def load_data(dataset="fashion_mnist"):
    path = f"../data/{dataset}/raw"
    
    # idx2numpy handles the magic numbers and headers for you automatically
    x_train = idx2numpy.convert_from_file(os.path.join(path, 'train-images-idx3-ubyte'))
    y_train_raw = idx2numpy.convert_from_file(os.path.join(path, 'train-labels-idx1-ubyte'))
    
    x_test = idx2numpy.convert_from_file(os.path.join(path, 't10k-images-idx3-ubyte'))
    y_test_raw = idx2numpy.convert_from_file(os.path.join(path, 't10k-labels-idx1-ubyte'))

    # Flatten and Normalize images to 0-1
    x_train = x_train.reshape(-1, 784).astype('float32')
    x_test = x_test.reshape(-1, 784).astype('float32') 

    # # One-hot encode
    # y_train = np.eye(10)[y_train_raw.astype(int)]
    # y_test = np.eye(10)[y_test_raw.astype(int)]

    return x_train, y_train_raw


def log_data_exploration():
    # 1. Initialize W&B run
    run = wandb.init(project="mnist_analysis", job_type="data_exploration")

    # 2. Load Dataset
    x_train, y_train_raw = load_data()
    class_names = [str(i) for i in range(10)]

    # 3. Create a W&B Table
    # Columns: Class Label and 5 columns for sample images
    columns = ["Digit Class"] + [f"Sample {i+1}" for i in range(5)]
    exploration_table = wandb.Table(columns=columns)

    # 4. Filter and Add Rows
    for class_id in range(10):
        # Find indices where the label matches the current digit (0-9)
        indices = np.where(y_train_raw == class_id)[0]
        
        # Pick 5 random indices for this specific digit
        selected_indices = np.random.choice(indices, 5, replace=False)
        
        row = [f"Digit {class_id}"]
        for idx in selected_indices:
            # FIX: Reshape the 1D array (784,) to 2D (28, 28) for W&B to render it
            img_2d = x_train[idx].reshape(28, 28)
            row.append(wandb.Image(img_2d))
        
        # Add the row of 5 images to the table
        exploration_table.add_data(*row)

    # 5. Log Table to W&B
    wandb.log({"MNIST Class Exploration": exploration_table})
    print("Successfully logged table to W&B.")
    run.finish()

if __name__ == "__main__":
    log_data_exploration()