import os, h5py
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from tqdm import tqdm
import copy, math
import collections

from cremerl import utils, model_zoo, shuffle

#import gymnasium as gym

import logging

# Set the logging level to WARNING
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)



expt_name = 'DeepSTARR'

# load data
data_path = '../../../data/'
filepath = os.path.join(data_path, expt_name+'_data.h5')
data_module = utils.H5DataModule(filepath, batch_size=100, lower_case=False, transpose=False)

deepstarr2 = model_zoo.deepstarr(2)
loss = torch.nn.MSELoss()
optimizer_dict = utils.configure_optimizer(deepstarr2, lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5, monitor='val_loss')
standard_cnn = model_zoo.DeepSTARR(deepstarr2,
                                  criterion=loss,
                                  optimizer=optimizer_dict)

# load checkpoint for model with best validation performance
standard_cnn = utils.load_model_from_checkpoint(standard_cnn, '../DeepSTARR_standard.ckpt')

# evaluate best model
pred = utils.get_predictions(standard_cnn, data_module.x_test[np.newaxis,100], batch_size=100)



def get_swap_greedy(x, x_mut, tile_ranges):
    ori = x.copy()
    mut = x_mut.copy()
    for tile_range in tile_ranges:
        ori[:, tile_range[0]:tile_range[1]] = x_mut[:, tile_range[0]:tile_range[1]]
        mut[:, tile_range[0]:tile_range[1]] = x[:, tile_range[0]:tile_range[1]]

    return ori, mut

def generate_tile_ranges(sequence_length, window_size, stride):
    ranges = []
    start = np.arange(0, sequence_length - window_size + stride, stride)

    for s in start:
        e = min(s + window_size, sequence_length)
        ranges.append([s, e])

    if start[-1] + window_size - stride < sequence_length:  # Adjust the last range
        ranges[-1][1] = sequence_length

    return ranges


def get_batch(x, tile_range, tile_ranges_ori, trials):
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

    #print(np.array(test_batch).shape)
    return np.array(test_batch)


def get_batch_score(pred, trials):

    score = []
    score_sep = []
    for i in range(0, pred.shape[0], 2):
        # print(f"Viewing number {i}")
        score1 = pred[0] - pred[i]
        score2 = pred[i+1] - pred[1]
        score.append((np.sum((score1, score2)[0])).tolist()) #np.sum(score1+score2, keepdims=True)
        score_sep.append((score1+score2).tolist())
        
    # print(score)
        
    final = np.sum(np.array(score), axis=0)/trials

    #max_ind = np.argmax(final)
    #block_ind = np.argmax(np.array(score)[:, max_ind])
    #print(np.array(total_score)[:, max_ind])
    total_score_sep = np.sum(np.array(score_sep), axis=0)/trials

    #print(np.max(score))
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


class SeqGame:
    def __init__(self, sequence, model_func):
        self.seq = sequence
        self.ori_seq = sequence.copy()
        self.tile_ranges = generate_tile_ranges(sequence.shape[1], 5, 5)
        self.levels = 20
        self.num_trials = 10
        self.action_size = 50
        
        self.prev_score = -float("inf")
        self.current_score = 0
        
        self.trainer = pl.Trainer(accelerator='gpu', devices='1', logger=None, enable_progress_bar=False)
        self.model = model_func
        
        if self.seq.shape[0]!=5:
            self.seq = extend_sequence(self.seq)
            self.ori_seq = extend_sequence(self.ori_seq)
        
    
    def get_initial_state(self):
        self.seq = self.ori_seq.copy()
        
        return self.seq
    
    
    def get_next_state(self, action, tile_ranges_done):
        self.prev_score = self.current_score
        # self.current_level += 1
        
        self.seq = taking_action(self.seq, self.tile_ranges[action])
        
        batch = get_batch(self.seq[:4, :], self.tile_ranges[action], tile_ranges_done, self.num_trials)
        dataloader = torch.utils.data.DataLoader(batch, batch_size=100, shuffle=False)
        pred = np.concatenate(self.trainer.predict(self.model, dataloaders=dataloader))
        
        # self.current_score = np.tanh(np.multiply(0.2, get_batch_score(pred, self.num_trials))) #ADDED TANH
        self.current_score = get_batch_score(pred, self.num_trials)
        
        return self.seq
    
    def get_valid_moves(self):
        return (convert_elements(self.seq[-1, :]) == 0).astype(np.uint8)
    
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
    
    
class Node:
    def __init__(self, action, state, done, reward, mcts, level, tile_ranges_done, parent=None):
        self.env = parent.env
        self.action = action
        
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        
        self.action_space_size = self.env.action_size
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32
        ) # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32) # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32
        ) # N
        self.valid_actions = (convert_elements(state[-1, :]) == 0).astype(np.bool_) # uint8
        
        self.reward = reward
        self.done = done
        self.state = state
        self.level = level
        
        self.tile_ranges_done = tile_ranges_done
        
        self.mcts = mcts
    
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]
    
    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value
        
    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value
        
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return (
            math.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )
    
    def best_action(self):
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        # masked_child_score = masked_child_score * self.valid_actions
        return np.argmax(masked_child_score)
    
    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node
    
    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors
        
    def set_state(self, state):
        self.state = state
        self.valid_actions = (convert_elements(state[-1, :]) == 0).astype(np.bool_) # uint8
    
    def get_child(self, action):
        if action not in self.children:

            self.env.set_seq(self.state.copy())
            next_state = self.env.get_next_state(action, self.tile_ranges_done)
            new_tile_ranges_done = copy.deepcopy(self.tile_ranges_done)
            new_tile_ranges_done.append(self.env.tile_ranges[action])
            # swap tile_ranges
            reward = self.env.get_score()
            terminated = self.env.terminate(self.level, reward, self.parent.reward)
            self.children[action] = Node(
                state=next_state, 
                action=action, 
                parent=self, 
                reward=reward,
                done=terminated,
                mcts=self.mcts, 
                level=self.level+1, 
                tile_ranges_done=new_tile_ranges_done
            )
        return self.children[action]
    
    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent

class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env
        self.reward = -np.inf


class CNN_v0(nn.Module):
    def __init__(self, action_dim):
        super(CNN_v0, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, action_dim) # 4 * action_dim
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, 128), 
            nn.Linear(128, 1), 
            # nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value



class CNN_v1(nn.Module):
    def __init__(self, action_dim, num_resBlocks=8):
        super(CNN_v1, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        
        self.resblocks = nn.ModuleList(
            [ResBlock(128) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, action_dim) # 4 * action_dim
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, 128), 
            nn.Linear(128, 1), 
            # nn.Tanh()
        )
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        for resBlock in self.resblocks:
            x = resBlock(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.conv2 = nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_hidden)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.activation(x)
        
        return x



class MCTS:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
    
    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model(torch.tensor(leaf.state).unsqueeze(0))
                child_priors = torch.softmax(child_priors, axis=1).squeeze(0).cpu().detach().numpy()
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size
                    )
                
                leaf.expand(child_priors)
            leaf.backup(value)
            
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(tree_policy)
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            action = np.argmax(tree_policy)
        else:
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]
    
    

class AlphaDNA:
    def __init__(self, model, optimizer, env, args, mcts_config = {
        "puct_coefficient": 2.0,
        "num_simulations": 10000,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": False,
        "add_dirichlet_noise": True,}
                 ):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.args = args
        self.mcts = MCTS(model, mcts_config)
        
        self.initial_sequence = env.get_seq()
        
        self.lowest_loss = np.inf
        self.period = 0
        
    def set_seq(self, seq):
        if seq.shape[0]!=5:
            seq = extend_sequence(seq)
            
        self.initial_sequence = seq
        self.env.set_seq(seq)
    
    def set_lr(self, learning_rate):
        self.optimizer.param_groups[0]['lr'] = learning_rate
        # param['lr'] = learning_rate
    
    def selfPlay(self):
        memory = []

        root_node = Node(
            state=self.initial_sequence,
            reward=0,
            done=False,
            action=None,
            parent=RootParentNode(env=self.env),
            mcts=self.mcts, 
            level=0, 
            tile_ranges_done=[]
        )

        while True:  # Loop until the root node indicates the game is done
            valid_moves = root_node.valid_actions
            mcts_probs, action, next_node = self.mcts.compute_action(root_node)
            
            memory.append((root_node.state, mcts_probs, next_node.reward))

            if valid_moves[action] == 0:
                print("Invalid action, skipping.")
                continue
            
            if next_node.done:
                return memory

            root_node = next_node
    
    def train(self, memory):
        np.random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets)
            
            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            
            out_policy, out_value = self.model(state)
            
            priors = nn.Softmax(dim=-1)(out_policy)
            
            policy_loss = torch.mean(
                -torch.sum(policy_targets * torch.log(priors), dim=-1)
            )
            value_loss = torch.mean(torch.pow(value_targets - out_value, 2))
            
            # print(out_value)
            
            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return total_loss
        
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            # print(f"Iteration {iteration}:")
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                print(f"Iteration {iteration} - SelfPlay Iteration {selfPlay_iteration}")
                memory += self.selfPlay()
            
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                loss = self.train(memory)
                print(f"Iteration {iteration} - Training Epoch {epoch+1} - Total Loss: {loss}")
                
                if loss < self.lowest_loss:
                    print("Saving the best model")
                    self.lowest_loss = loss
                    self.period = 0
                    torch.save(self.model.state_dict(), "best_model.pt")
                    torch.save(self.optimizer.state_dict(), "best_optimizer.pt")
                
                if self.optimizer.param_groups[0]['lr'] <= self.args['rlop_minimum']:
                        return True
                
                if self.period > self.args['rlop_patience']:
                    self.period = 0
                    lr = self.optimizer.param_groups[0]['lr']
                    lr *= self.args['rlop_factor']
                    self.set_lr(lr)
                    print(f"Reducing learning rate to {lr}")
                self.period += 1
            
        
        torch.save(self.model.state_dict(), "best_iteration_model.pt")
        torch.save(self.optimizer.state_dict(), "best_iteration_optimizer.pt")
        return False
            
            


seqgame = SeqGame(data_module.x_train[1].numpy(), standard_cnn)
# model = CNN_v0(seqgame.action_size)
model = CNN_v1(seqgame.action_size, num_resBlocks=3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'num_iterations': 1, # 3 
    'num_selfPlay_iterations': 4, # 500
    'num_epochs': 4, 
    'batch_size': 4, # 64
    'rlop_patience': 80,
    'rlop_factor': 0.3,
    'rlop_minimum': 1e-7,
}

mcts_config = {
    "puct_coefficient": 2.0, # 2.0
    "num_simulations": 1000,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": True,
}

alphadna = AlphaDNA(model, optimizer, seqgame, args, mcts_config)

for _ in tqdm(range(10)):
    for sequence in data_module.x_test[0:100]:
        # sequence = data_module.x_train[1].numpy()
        seq = sequence.numpy()
        
        alphadna.set_seq(seq)
        early = alphadna.learn()
        if early:
            print("Early Stopping")
            break
    else:
        continue
    break