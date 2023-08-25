import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
from tqdm import tqdm

from cremerl import utils, model_zoo
import env, alphadna, models

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
# pred = utils.get_predictions(standard_cnn, data_module.x_test[np.newaxis,100], batch_size=100)



seqgame = env.SeqGame(data_module.x_train[1].numpy(), standard_cnn, num_trials=10)
# model = CNN_v0(seqgame.action_size)
model = models.CNN_v1(seqgame.action_size, num_resBlocks=3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'num_iterations': 5, # 3 
    'num_selfPlay_iterations': 1, # 500
    'num_parallel_games': 1, # 10
    'num_epochs': 1, # 4 
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

alphadna = alphadna.AlphaDNA_P(model, optimizer, seqgame, args, mcts_config)

for _ in tqdm(range(10)):
    for sequence in data_module.x_test[0:1000]:
        # sequence = data_module.x_train[1].numpy()
        seq = sequence.numpy()
        
        alphadna.set_seq(seq)
        early = alphadna.learn()
        
        # alphadna.model.load_state_dict(torch.load("best_model.pt"))
        
    #     if early:
    #         print("Early Stopping")
    #         break
    # else:
    #     continue
    # break