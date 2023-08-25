import numpy as np
import pytorch_lightning as pl
import torch
import utils

class SeqGame:
    def __init__(self, sequence, model_func, num_trials=10):
        # Initialize a sequential game environment
        self.seq = sequence  # Current sequence
        self.ori_seq = sequence.copy()  # Original sequence for reset
        self.tile_ranges = utils.generate_tile_ranges(sequence.shape[1], 5, 5)  # Generate tile ranges for actions
        self.levels = 20  # Maximum levels
        self.num_trials = num_trials  # Number of trials for batch evaluation
        self.action_size = 50  # Size of the action space
        
        self.prev_score = -float("inf")  # Previous score for comparison
        self.current_score = 0  # Current score
        
        self.trainer = pl.Trainer(accelerator='gpu', devices='1', logger=None, enable_progress_bar=False)  # PyTorch Lightning trainer
        self.model = model_func  # Model function
        
        if self.seq.shape[0] != 5:
            self.seq = utils.extend_sequence(self.seq)  # Extend sequence if needed
            self.ori_seq = utils.extend_sequence(self.ori_seq)
        
    def get_initial_state(self):
        # Reset the sequence to its original state
        self.seq = self.ori_seq.copy()
        return self.seq
    
    def get_next_state(self, action, tile_ranges_done):
        # Get the next state after taking an action
        self.prev_score = self.current_score
        
        # Apply action and obtain the next sequence
        self.seq = utils.taking_action(self.seq, self.tile_ranges[action])
        
        # Prepare batch for model evaluation
        batch = utils.get_batch(self.seq[:4, :], self.tile_ranges[action], tile_ranges_done, self.num_trials)
        dataloader = torch.utils.data.DataLoader(batch, batch_size=100, shuffle=False)
        
        # Predict using the model and concatenate results
        pred = np.concatenate(self.trainer.predict(self.model, dataloaders=dataloader))
        
        # Update the current score based on batch predictions
        self.current_score = utils.get_batch_score(pred, self.num_trials)
        
        return self.seq
    
    def get_valid_moves(self):
        # Get a mask of valid moves based on the current sequence
        return (utils.convert_elements(self.seq[-1, :]) == 0).astype(np.uint8)
    
    def terminate(self, level, current_score, parent_score):
        # Determine if the episode should terminate
        if level >= self.levels:
            return True
        if current_score < parent_score:
            return True
        return False
    
    def set_seq(self, seq):
        # Set the current sequence
        self.seq = seq
    
    def get_seq(self):
        # Get a copy of the current sequence
        return self.seq.copy()
    
    def get_score(self):
        # Get the current score
        return self.current_score
