POINT_NORMALIZATION = 36.4 #normalize by point change in round that yields std.dev of 1
LEARNING_RATE = 0.001

######################### DQN PARAMETERS ##############################
#### REPLAY BUFFER PARAMETERS
DQN_ACTION_REPLAY_BUFFER_SIZE = 250000 #equivalent of ~350 games
DQN_BET_REPLAY_BUFFER_SIZE    = 35000
DQN_EVAL_REPLAY_BUFFER_SIZE   = 65000

#### RETRAINING
RETRAIN_BATCH_SIZE = 676
RETRAIN_EPOCHS = 1

#### EVAL AND BET TRAINING
DQN_NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN = 32000
DQN_EVAL_TRAIN_EPOCHS = 128
DQN_EVAL_TRAIN_BATCH_SIZE = 256
DQN_BET_TRAIN_EPOCHS = 128
DQN_BET_TRAIN_BATCH_SIZE = 256
#######################################################################

######################### A3C PARAMETERS ##############################
A3C_NUM_GAMES_PER_WORKER = 1
A3C_NUM_WORKERS = 1
A3C_GLOBAL_NET_UPDATE_EVAL_FREQ = 1
#######################################################################

#################### MODEL COMPARISON PARAMETERS ######################
ITER_WOUT_IMPROVE_BEFORE_RESET = 3
COMPARISON_GAMES = 50

CONFIDENCE_REQ = 0.95
FURTHER_EXPLORE_CONFIDENCE_REQ = 0.75