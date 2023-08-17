import torch
import torch.nn as nn
import numpy as np
import utils
import mcts


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
        self.mcts = mcts.MCTS(model, mcts_config)
        
        self.initial_sequence = env.get_seq()
        
        self.lowest_loss = np.inf
        self.period = 0
        
    def set_seq(self, seq):
        if seq.shape[0]!=5:
            seq = utils.extend_sequence(seq)
            
        self.initial_sequence = seq
        self.env.set_seq(seq)
    
    def set_lr(self, learning_rate):
        self.optimizer.param_groups[0]['lr'] = learning_rate
        # param['lr'] = learning_rate
    
    def selfPlay(self):
        memory = []

        root_node = mcts.Node(
            state=self.initial_sequence,
            reward=0,
            done=False,
            action=None,
            parent=mcts.RootParentNode(env=self.env),
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
    
    def predict(self, sequence):
        root_node = mcts.Node(
            state=sequence,
            reward=0,
            done=False,
            action=None,
            parent=mcts.RootParentNode(env=self.env),
            mcts=self.mcts, 
            level=0, 
            tile_ranges_done=[]
        )

        while True:  # Loop until the root node indicates the game is done
            mcts_probs, action, next_node = self.mcts.compute_action(root_node)

            if next_node.done:
                return root_node.tile_ranges_done, root_node.reward

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
                # print(f"SelfPlay Iteration {selfPlay_iteration}")
                memory += self.selfPlay()
            
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                loss = self.train(memory)
                print(f"Training Epoch {epoch+1} - Total Loss: {loss}")
                
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
            
            
            

class AlphaDNA_P:
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
        self.mcts = mcts.MCTS_P(model, mcts_config)
        
        self.initial_sequence = env.get_seq()

    def set_seq(self, seq):
        if seq.shape[0]!=5:
            seq = utils.extend_sequence(seq)
            
        self.initial_sequence = seq
        self.env.set_seq(seq)
    
    def set_lr(self, learning_rate):
        self.optimizer.param_groups[0]['lr'] = learning_rate
        # param['lr'] = learning_rate
        
    def selfPlay(self):
        memory = []

        root_node = mcts.Node(
            state=self.initial_sequence,
            reward=0,
            done=False,
            action=None,
            parent=mcts.RootParentNode(env=self.env),
            mcts=self.mcts, 
            level=0, 
            tile_ranges_done=[]
        )
        
        buffers = [Buffer(root_node) for buffer in range(self.args['num_parallel_games'])]
        
        while len(buffers) > 0:
            self.mcts.compute_action(buffers) 
            
            for i in range(len(buffers))[::-1]:
                buffer = buffers[i]
                
                buffer.memory.append((buffer.root.state, buffer.tree_policy, buffer.next_node.reward))
                
                if buffer.next_node.done:
                    for hist_states, hist_probs, hists_reward in buffer.memory:
                        memory.append((hist_states, hist_probs, hists_reward))
                    
                    del buffers[i]
                
                buffer.root = buffer.next_node
        
        return memory
                        
    
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
                
                # if self.optimizer.param_groups[0]['lr'] <= self.args['rlop_minimum']:
                #         return True
                
                # if self.period > self.args['rlop_patience']:
                #     self.period = 0
                #     lr = self.optimizer.param_groups[0]['lr']
                #     lr *= self.args['rlop_factor']
                #     self.set_lr(lr)
                #     print(f"Reducing learning rate to {lr}")
                # self.period += 1
            
        
        torch.save(self.model.state_dict(), "iteration_model.pt")
        torch.save(self.optimizer.state_dict(), "iteration_optimizer.pt")
        return False
                
            
class Buffer:
    def __init__(self, node):
        self.memory = []
        self.root = node
        self.node = None
        self.tree_policy = None
        self.action = None
        self.next_node = None