import os
import nn_config
import argparse
import random
import time
import tensorflow as tf
from JudgmentGame import JudgmentGame
from NNAgent import NNAgent
from compare_agents import compareAgents
import multiprocessing
import wandb
import numpy as np
from copy import copy, deepcopy
from multiprocessing import cpu_count, Pool
from judgment_value_models import initBetModel, initEvalModel, initActionModel
from agent_training_utils import loadModels, evaluateModelPerformance
from judgment_data_utils import postProcessBetTrainData, postProcessTrainData, convertSubroundSituationToActionState
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#GPUs in general do not play nicely with multiprocessing, so we disable them here.
#Also, performance did not seem to be improved by using GPUs, so we're not losing much.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _build_parser():
    parser = argparse.ArgumentParser(description='Run NN to generate ideal playing strategy for SPLT.')

    parser.add_argument(
        '-r','--run_name',
        help="Will name the output directory for the results of the run",
        type=str,
        default="test_run",
    )

    parser.add_argument(
        '-b','--bet_model_path',
        help="Path from current directory to folder containing bet model",
        type=str,
        default="bet_expert_train_model",
    )

    parser.add_argument(
        '-e','--eval_model_path',
        help="Path from current directory to folder containing eval model",
        type=str,
        default="eval_expert_train_model",
    )

    parser.add_argument(
        '-a','--action_model_path',
        help="Path from current directory to folder containing action model",
        type=str,
        default="action_expert_train_model",
    )

    parser.add_argument(
        '-m','--models_path',
        help="Path to folder containing models, if you're restarting from a previous run. Defaults to None, in which case will use -b, -e, and -a to find models.",
        default=None,
    )

    parser.add_argument(
        '-t','--track',
        help="Flag determining whether to track this run in weights and biases",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        '--wandb_project_name',
        help="Name of WandB project to log to. Only relevant if args.track=True.",
        type=str,
        default="JudgementBot",
    )

    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default="josh-holder", 
        help="the entity (team) of wandb's project"
    )

    return parser

def initWandBTrack(args):
    #initialize weights and biases tracking
    run_name = f"{args.run_name}__{int(time.time())}"
    config_dict = vars(copy(args))
    config_dict.pop("track",None)
    config_dict.pop("wandb_project_name",None)
    config_dict.pop("wandb_entity",None)
    config_dict["algo"] = "a3c"
    
    nn_config_dict = nn_config.__dict__
    for key in list(nn_config.__dict__.keys()):
        if key.startswith("_"): nn_config_dict.pop(key,None)
        if key.startswith("DQN"): nn_config_dict.pop(key,None)

    config_dict = {**config_dict, **nn_config.__dict__}
    wandb.init(
        name=run_name,
        entity=args.wandb_entity,
        project=args.wandb_project_name,
        config=config_dict,
    )

def playJudgmentGameThread(core_id, curr_action_weights, curr_bet_weights, curr_eval_weights, epsilon_choice):
    """
    Plays a game of Judgment with an asynchronous actor, given models and a randomly chosen epsilon value.

    At the end of each round of the game, the actor will compute a gradient update of its current weights.
    Along the way, it accumulates the gradients for the action, bet, and eval models, eventually returning them
    to be applied to the global weight network.

    It also keeps track of how many training examples it has seen, which is returned to be tracked by the main thread.
    """
    thread_bet_model = initBetModel()
    thread_bet_model.set_weights(curr_bet_weights)

    thread_eval_model = initEvalModel()
    thread_eval_model.set_weights(curr_eval_weights)

    thread_action_model = initActionModel()
    thread_action_model.set_weights(curr_action_weights)

    accum_act_gradients = [tf.zeros_like(var) for var in thread_action_model.trainable_variables]
    accum_bet_gradients = [tf.zeros_like(var) for var in thread_bet_model.trainable_variables]
    accum_eval_gradients = [tf.zeros_like(var) for var in thread_eval_model.trainable_variables]

    act_training_examples = 0
    bet_training_examples = 0
    eval_training_examples = 0

    for game_num in range(nn_config.A3C_NUM_GAMES_PER_WORKER):
        start = time.time()
        #Initialize game
        jg = JudgmentGame([NNAgent(0,epsilon_choice,load_models=False), NNAgent(1,epsilon_choice,load_models=False), \
                           NNAgent(2,epsilon_choice,load_models=False), NNAgent(3,epsilon_choice,load_models=False)])
        
        #Set models for each agent
        for agent in jg.agents:
            agent.action_model = thread_action_model
            agent.bet_model = thread_bet_model
            agent.eval_model = thread_eval_model

        bet_train_data, eval_train_data, action_train_data = jg.playGameAndCollectData()

        #Compute loss on action model
        with tf.GradientTape() as action_tape:
            action_data_inputs = []
            action_data_outputs = []
            for action_data in action_train_data:
                action_data_inputs.append(action_data[0])
                action_data_outputs.append(action_data[1])

            action_data_inputs = postProcessTrainData(action_data_inputs)
            action_data_outputs = np.array(action_data_outputs)

            action_predictions = thread_action_model(action_data_inputs)

            action_loss = tf.keras.losses.MeanSquaredError()
            action_losses = action_loss(action_data_outputs, action_predictions)

            action_gradients = action_tape.gradient(action_losses, thread_action_model.trainable_variables)

        #Compute loss on bet model
        with tf.GradientTape() as bet_tape:
            #split the bet_data from tuple form (input, output) to separate lists
            bet_data_inputs = []
            bet_data_outputs = []
            for bet_data in bet_train_data:
                bet_data_inputs.append(bet_data[0])
                bet_data_outputs.append(bet_data[1])
            
            bet_data_inputs = postProcessBetTrainData(bet_data_inputs)
            bet_data_outputs = np.array(bet_data_outputs)
            
            bet_predictions = thread_bet_model(bet_data_inputs)

            bet_loss = tf.keras.losses.MeanSquaredError()
            bet_losses = bet_loss(bet_data_outputs, bet_predictions)

            bet_gradients = bet_tape.gradient(bet_losses, thread_bet_model.trainable_variables)

        #Compute loss on eval model
        with tf.GradientTape() as eval_tape:
            eval_data_inputs = []
            eval_data_outputs = []
            for eval_data in eval_train_data:
                eval_data_inputs.append(eval_data[0])
                eval_data_outputs.append(eval_data[1])

            eval_data_inputs = postProcessTrainData(eval_data_inputs)
            eval_data_outputs = np.array(eval_data_outputs, dtype='float32')

            eval_predictions = thread_eval_model(eval_data_inputs)
            eval_loss = tf.keras.losses.MeanSquaredError()
            eval_losses = eval_loss(eval_data_outputs, eval_predictions)

            eval_gradients = eval_tape.gradient(eval_losses, thread_eval_model.trainable_variables)

        #Accumulate gradients
        accum_act_gradients = accum_act_gradients + action_gradients
        accum_bet_gradients = accum_bet_gradients + bet_gradients
        accum_eval_gradients = accum_eval_gradients + eval_gradients

        act_training_examples += len(action_predictions)
        bet_training_examples += len(bet_predictions)
        eval_training_examples += len(eval_predictions)

        print(f"Game {game_num+1}/{nn_config.A3C_NUM_GAMES_PER_WORKER} on core {core_id} took {time.time()-start} seconds.", end='\r')

    return accum_act_gradients, accum_bet_gradients, accum_eval_gradients,\
            act_training_examples, bet_training_examples, eval_training_examples


def trainAgentViaA3C():
    #Without this, was having issues with initializing LSTM layers in subprocesses.
    #https://github.com/keras-team/keras/issues/10095
    multiprocessing.set_start_method('spawn')

    parser = _build_parser()
    args = parser.parse_args()

    if args.track: initWandBTrack(args)

    run_folder_path = os.path.join(os.getcwd(),args.run_name)
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)

    curr_bet_model, curr_eval_model, curr_action_model,\
        best_bet_model, best_eval_model, best_action_model,\
            baseline_bet_model, baseline_eval_model, baseline_action_model = loadModels(args)

    state_action_examples_trained_on = 0
    bet_examples_trained_on = 0

    epsilon_choices = [0.3, 0.35, 0.25, 0.2, 0.15, 0.1]
    num_global_updates = 0

    iterations_without_improving = 0

    print("~~~BEGINNING TRAINING~~~")
    while True:
        #Initialize gradients of zeros to eventually apply to the global network
        act_gradients = [tf.zeros_like(var) for var in curr_action_model.trainable_variables]
        bet_gradients = [tf.zeros_like(var) for var in curr_bet_model.trainable_variables]
        eval_gradients = [tf.zeros_like(var) for var in curr_eval_model.trainable_variables]

        #init arguments for each worker
        game_arguments = []
        for worker in range(nn_config.A3C_NUM_WORKERS):
            epsilon_choice = random.choice(epsilon_choices)
            game_arguments.append((worker, curr_action_model.get_weights(), curr_bet_model.get_weights(), curr_eval_model.get_weights(), epsilon_choice))

        #Initialize a pool of processes to play games and accumulate gradients, each using a random choice of epsilon
        print(f"Spawning {nn_config.A3C_NUM_WORKERS} workers to play {nn_config.A3C_NUM_GAMES_PER_WORKER} games each and accumulate gradients:")
        with Pool(processes=nn_config.A3C_NUM_WORKERS) as p:
            worker_accum_gradients = p.starmap(playJudgmentGameThread, game_arguments)

        #Accumulate gradients from each worker
        for (worker_accum_act_gradients, worker_accum_bet_gradients, worker_accum_eval_gradients,\
             worker_act_training_examples, worker_bet_training_examples, worker_eval_training_examples) in worker_accum_gradients:
            act_gradients = act_gradients + worker_accum_act_gradients
            bet_gradients = bet_gradients + worker_accum_bet_gradients
            eval_gradients = eval_gradients + worker_accum_eval_gradients

            state_action_examples_trained_on += worker_act_training_examples
            bet_examples_trained_on += worker_bet_training_examples

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=nn_config.LEARNING_RATE)

        #update weights by the accumulated gradients
        optimizer.apply_gradients(zip(act_gradients, curr_action_model.trainable_variables))
        optimizer.apply_gradients(zip(bet_gradients, curr_bet_model.trainable_variables))
        optimizer.apply_gradients(zip(eval_gradients, curr_eval_model.trainable_variables))

        num_global_updates += 1

        print(f"\nApplied global gradient update {num_global_updates}!")

        if args.track:
            wandb.log({"train/state_action_examples_trained_on": state_action_examples_trained_on,
                       "train/bet_examples_trained_on": bet_examples_trained_on,
                       "train/global_gradient_updates": num_global_updates})

        if num_global_updates % nn_config.A3C_GLOBAL_NET_UPDATE_EVAL_FREQ == 0:
            curr_action_model, curr_bet_model, curr_eval_model, \
            best_action_model, best_bet_model, best_eval_model, \
            baseline_action_model, baseline_bet_model, baseline_eval_model, \
            iterations_without_improving = evaluateModelPerformance(curr_action_model, curr_bet_model, curr_eval_model,\
                                    best_action_model, best_bet_model, best_eval_model,\
                                    baseline_action_model, baseline_bet_model, baseline_eval_model,\
                                    iterations_without_improving, args)

if __name__ == "__main__":
    trainAgentViaA3C()