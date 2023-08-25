import numpy as np
import pytorch_lightning as pl
import torch
import utils


class SeqGame:
    def __init__(self, sequence, model_func, num_trials=10):
        self.seq = sequence
        self.ori_seq = sequence.copy()
        self.tile_ranges = utils.generate_tile_ranges(sequence.shape[1], 5, 5)
        self.levels = 20
        self.num_trials = num_trials
        self.action_size = 50
        
        self.prev_score = -float("inf")
        self.current_score = 0
        
        self.trainer = pl.Trainer(accelerator='gpu', devices='1', logger=None, enable_progress_bar=False)
        self.model = model_func
        
        if self.seq.shape[0]!=5:
            self.seq = utils.extend_sequence(self.seq)
            self.ori_seq = utils.extend_sequence(self.ori_seq)
        
    
    def get_initial_state(self):
        self.seq = self.ori_seq.copy()
        
        return self.seq
    
    
    def get_next_state(self, action, tile_ranges_done):
        self.prev_score = self.current_score
        # self.current_level += 1
        
        self.seq = utils.taking_action(self.seq, self.tile_ranges[action])
        
        batch = utils.get_batch(self.seq[:4, :], self.tile_ranges[action], tile_ranges_done, self.num_trials)
        dataloader = torch.utils.data.DataLoader(batch, batch_size=100, shuffle=False)
        pred = np.concatenate(self.trainer.predict(self.model, dataloaders=dataloader))
        
        # self.current_score = np.tanh(np.multiply(0.2, get_batch_score(pred, self.num_trials))) #ADDED TANH
        self.current_score = utils.get_batch_score(pred, self.num_trials)
        
        return self.seq
    
    def get_valid_moves(self):
        return (utils.convert_elements(self.seq[-1, :]) == 0).astype(np.uint8)
    
    def terminate(self, level, current_score, parent_score): #state
        # if self.current_level >= self.levels:
        #     return True
        # if self.current_score < self.prev_score:
        #     return True
        
        if level >= self.levels:
            return True
        if current_score < parent_score:
            return True
    
        return False
    
    def set_seq(self, seq):
        self.seq = seq
    
    def get_seq(self):
        return self.seq.copy()
    
    def get_score(self):
        return self.current_score