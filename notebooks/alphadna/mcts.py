import numpy as np
import torch
import copy, math
import collections

import utils

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
        self.valid_actions = (utils.convert_elements(state[-1, :]) == 0).astype(np.bool_) # uint8
        
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
        self.valid_actions = (utils.convert_elements(state[-1, :]) == 0).astype(np.bool_) # uint8
    
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
    

class MCTS_P:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
    
    def compute_action(self, buffers):
        for _ in range(self.num_sims):
            for buffer in buffers:
                leaf = buffer.root.select()
                if leaf.done:
                    buffer_value = leaf.reward
                    leaf.backup(buffer_value)
                else:
                    buffer.node = leaf
        
            expandable_buffers = [mappingIdx for mappingIdx in range(len(buffers)) if buffers[mappingIdx].node.done is False]
            
            
            if len(expandable_buffers) > 0:
                states = np.stack([buffers[mappingIdx].node.state for mappingIdx in expandable_buffers])
                child_priors, values = self.model(torch.tensor(states))
                child_priors = torch.softmax(child_priors, axis=1).cpu().detach().numpy()
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * buffer.node.action_space_size # child_priors.size
                    , size=child_priors.shape[0])
            
            for i, mappingIdx in enumerate(expandable_buffers):
                node = buffers[mappingIdx].node
                buffer_child_priors, buffer_value = child_priors[i], values[i]
                
                node.expand(buffer_child_priors)
                node.backup(buffer_value)
        
        for buffer in buffers:
            node = buffer.root
            tree_policy = node.child_number_visits / node.number_visits
            tree_policy = tree_policy / np.max(tree_policy)
            tree_policy = np.power(tree_policy, self.temperature)
            tree_policy = tree_policy / np.sum(tree_policy)
            if self.exploit:
                action = np.argmax(tree_policy)
            else:
                action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
            
            buffer.tree_policy = tree_policy
            buffer.action = action
            buffer.next_node = node.children[action]
            