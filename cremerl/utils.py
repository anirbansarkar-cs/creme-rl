# Credits: This script is taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
import os, pathlib, h5py
import numpy as np
from scipy import stats
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from . import shuffle



def get_swap_greedy(x, x_mut, tile_ranges):
    """
    Generate two sequences by swapping tiles according to tile_ranges.

    Parameters:
    - x (numpy.ndarray): The original sequence.
    - x_mut (numpy.ndarray): The mutated sequence.
    - tile_ranges (list): List of tile ranges for swapping.

    Returns:
    - ori (numpy.ndarray): The original sequence with tile swaps.
    - mut (numpy.ndarray): The mutated sequence with tile swaps.
    """
    ori = x.copy()
    mut = x_mut.copy()
    for tile_range in tile_ranges:
        ori[:, tile_range[0]:tile_range[1]] = x_mut[:, tile_range[0]:tile_range[1]]
        mut[:, tile_range[0]:tile_range[1]] = x[:, tile_range[0]:tile_range[1]]
    return ori, mut

def generate_tile_ranges(sequence_length, window_size, stride):
    """
    Generate tile ranges for sliding window.

    Parameters:
    - sequence_length (int): Length of the sequence.
    - window_size (int): Size of the sliding window.
    - stride (int): Stride for the sliding window.

    Returns:
    - ranges (list): List of tile ranges.
    """
    ranges = []
    start = np.arange(0, sequence_length - window_size + stride, stride)
    for s in start:
        e = min(s + window_size, sequence_length)
        ranges.append([s, e])
    if start[-1] + window_size - stride < sequence_length:  # Adjust the last range
        ranges[-1][1] = sequence_length
    return ranges

def get_batch(x, tile_range, tile_ranges_ori, trials):
    """
    Generate a batch of sequences with tile swaps for evaluation.

    Parameters:
    - x (numpy.ndarray): The input sequence.
    - tile_range (list): Tile range for current action.
    - tile_ranges_ori (list): Tile ranges for original sequence.
    - trials (int): Number of trials for generating the batch.

    Returns:
    - test_batch (numpy.ndarray): Batch of sequences with tile swaps.
    """
    test_batch = []
    for i in range(trials):
        test_batch.append(x)
        x_mut = shuffle.dinuc_shuffle(x.copy())
        test_batch.append(x_mut)

        ori = x.copy()
        mut = x_mut.copy()
        
        ori, mut = get_swap_greedy(ori, mut, tile_ranges_ori)
        
        ori[:, tile_range[0]:tile_range[1]] = x_mut[:, tile_range[0]:tile_range[1]]
        mut[:, tile_range[0]:tile_range[1]] = x[:, tile_range[0]:tile_range[1]]
        
        test_batch.append(ori)
        test_batch.append(mut)
        
    return np.array(test_batch)

def get_batch_score(pred, trials):
    """
    Calculate scores for a batch of predictions.

    Parameters:
    - pred (numpy.ndarray): Batch of prediction results.
    - trials (int): Number of trials for generating the batch.

    Returns:
    - final (numpy.ndarray): Array of calculated scores.
    """
    score = []
    score_sep = []
    for i in range(0, pred.shape[0], 2):
        score1 = pred[0] - pred[i]
        score2 = pred[i+1] - pred[1]
        score.append((np.sum((score1, score2)[0])).tolist())
        score_sep.append((score1+score2).tolist())
        
    final = np.sum(np.array(score), axis=0) / trials
    total_score_sep = np.sum(np.array(score_sep), axis=0) / trials

    return final


def extend_sequence(one_hot_sequence):
    A, L = one_hot_sequence.shape

    # Create an all-ones row
    ones_row = np.zeros(L)

    # Add the all-ones row to the original sequence
    new_sequence = np.vstack((one_hot_sequence, ones_row))

    return np.array(new_sequence, dtype='float32')

def taking_action(sequence_with_ones, tile_range):
    start_idx, end_idx = tile_range

    # Ensure the start_idx and end_idx are within valid bounds
    #if start_idx < 0 or start_idx >= sequence_with_ones.shape[1] or end_idx < 0 or end_idx >= sequence_with_ones.shape[1]:
    #    raise ValueError("Invalid tile range indices.")

    # Copy the input sequence to avoid modifying the original sequence
    modified_sequence = sequence_with_ones.copy()

    # Modify the last row within the specified tile range
    modified_sequence[-1, start_idx:end_idx] = 1

    return np.array(modified_sequence, dtype='float32')


def convert_elements(input_list):
    input_list = input_list.tolist()
    num_columns = 5  # Number of elements to process in each group

    # Calculate the number of elements needed to pad the list
    padding_length = num_columns - (len(input_list) % num_columns)
    last_value = input_list[-1]
    padded_list = input_list + [last_value] * padding_length

    # Convert the padded list to a NumPy array for efficient operations
    input_array = np.array(padded_list)
    reshaped_array = input_array.reshape(-1, num_columns)

    # Check if each row has the same value (all 0s or all 1s)
    row_all_zeros = np.all(reshaped_array == 0, axis=1)
    row_all_ones = np.all(reshaped_array == 1, axis=1)

    # Replace all 0s with 0 and all 1s with 1 in the result array
    output_array = np.where(row_all_zeros, 0, np.where(row_all_ones, 1, reshaped_array[:, 0]))

    # Flatten the result array to get the final output list
    output_list = output_array.flatten()

    return output_list




def load_model_from_checkpoint(model, checkpoint_path):
    """Load PyTorch lightning model from checkpoint."""
    return model.load_from_checkpoint(checkpoint_path,
        model=model.model,
        criterion=model.criterion,
        optimizer=model.optimizer,
    )


def configure_optimizer(model, lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5, monitor='val_loss'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, patience=patience),
            "monitor": monitor,
        },
    }
    

def get_predictions(model, x, batch_size=100, accelerator='gpu', devices=1):
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=None)
    dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False) 
    pred = trainer.predict(model, dataloaders=dataloader)
    return np.concatenate(pred)


def evaluate_model(y_test, pred):
    pearsonr = calculate_pearsonr(y_test, pred)
    spearmanr = calculate_spearmanr(y_test, pred)
    #print("Test Pearson r : %.4f +/- %.4f"%(np.nanmean(pearsonr), np.nanstd(pearsonr)))
    #print("Test Spearman r: %.4f +/- %.4f"%(np.nanmean(spearmanr), np.nanstd(spearmanr)))
    print("  Pearson r: %.4f \t %.4f"%(pearsonr[0], pearsonr[1]))
    print("  Spearman : %.4f \t %.4f"%(spearmanr[0], spearmanr[1]))
    return pearsonr, spearmanr

def calculate_pearsonr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)
    
def calculate_spearmanr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)




class H5DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, stage=None, lower_case=False, transpose=False, downsample=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.x = 'X'
        self.y = 'Y'
        if lower_case:
            self.x = 'x'
            self.y = 'y'
        self.transpose = transpose
        self.downsample = downsample
        self.setup(stage)

    def setup(self, stage=None):
        # Assign train and val split(s) for use in DataLoaders
        if stage == "fit" or stage is None:
            with h5py.File(self.data_path, 'r') as dataset:
                x_train = np.array(dataset[self.x+"_train"]).astype(np.float32)
                y_train = np.array(dataset[self.y+"_train"]).astype(np.float32)
                x_valid = np.array(dataset[self.x+"_valid"]).astype(np.float32)
                if self.transpose:
                    x_train = np.transpose(x_train, (0,2,1))
                    x_valid = np.transpose(x_valid, (0,2,1))
                if self.downsample:
                    x_train = x_train[:self.downsample]
                    y_train = y_train[:self.downsample]
                self.x_train = torch.from_numpy(x_train)
                self.y_train = torch.from_numpy(y_train)
                self.x_valid = torch.from_numpy(x_valid)
                self.y_valid = torch.from_numpy(np.array(dataset[self.y+"_valid"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape # N = number of seqs, A = alphabet size (number of nucl.), L = length of seqs
            self.num_classes = self.y_train.shape[1]
            
        # Assign test split(s) for use in DataLoaders
        if stage == "test" or stage is None:
            with h5py.File(self.data_path, "r") as dataset:
                x_test = np.array(dataset[self.x+"_test"]).astype(np.float32)
                if self.transpose:
                    x_test = np.transpose(x_test, (0,2,1))
                self.x_test = torch.from_numpy(x_test)
                self.y_test = torch.from_numpy(np.array(dataset[self.y+"_test"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape
            self.num_classes = self.y_train.shape[1]
            
    def train_dataloader(self):
        train_dataset = TensorDataset(self.x_train, self.y_train) # tensors are index-matched
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # sets of (x, x', y) will be shuffled
    
    def val_dataloader(self):
        valid_dataset = TensorDataset(self.x_valid, self.y_valid) 
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 
