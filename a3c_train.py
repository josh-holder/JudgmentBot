import os
import nn_config
import argparse
import numpy as np
import random
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from JudgmentDataUtils import postProcessTrainData, postProcessBetTrainData, convertSubroundSituationToActionState
from JudgmentGame import JudgmentGame
from DQNAgent import DQNAgent
from compare_agents import compareAgents
from HumanBetAgent import HumanBetAgent
import multiprocessing
import wandb
from copy import copy, deepcopy
from multiprocessing import cpu_count, Pool
from JudgmentValueModels import initBetModel, initEvalModel, initActionModel
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#GPUs in general do not play nicely with multiprocessing, so we disable them here.
#Also, performance did not seem to be improved by using GPUs, so we're not losing much.
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
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

def loadModels(args):
    """
    Given command line arguments, loads current, baseline, and target models.
    """
    import keras
    print("loading models")
    if args.models_path == None:
        curr_bet_model_path = os.path.join(os.getcwd(),args.bet_model_path)
        curr_bet_model = keras.models.load_model(curr_bet_model_path, compile=False)
        print(f"Loaded current bet model from {curr_bet_model_path}")

        curr_eval_model_path = os.path.join(os.getcwd(),args.eval_model_path)
        curr_eval_model = keras.models.load_model(curr_eval_model_path, compile=False)
        print(f"Loaded current eval model from {curr_eval_model_path}")

        curr_action_model_path = os.path.join(os.getcwd(),args.action_model_path)
        curr_action_model = keras.models.load_model(curr_action_model_path, compile=False)
        print(f"Loaded current action model fron {curr_action_model_path}")
    
    else:
        curr_bet_model_path = os.path.join(os.getcwd(),args.models_path,"best_bet_model")
        curr_bet_model = keras.models.load_model(curr_bet_model_path, compile=False)
        print(f"Loaded current bet model from {curr_bet_model_path}")

        curr_eval_model_path = os.path.join(os.getcwd(),args.models_path,"best_eval_model")
        curr_eval_model = keras.models.load_model(curr_eval_model_path, compile=False)
        print(f"Loaded current eval model from {curr_eval_model_path}")

        curr_action_model_path = os.path.join(os.getcwd(),args.models_path,"best_act_model")
        curr_action_model = keras.models.load_model(curr_action_model_path, compile=False)
        print(f"Loaded current action model fron {curr_action_model_path}")

    baseline_bet_model = tf.keras.models.clone_model(curr_bet_model)
    baseline_bet_model.set_weights(curr_bet_model.get_weights())
    baseline_bet_model.compile()

    baseline_eval_model = tf.keras.models.clone_model(curr_eval_model)
    baseline_eval_model.set_weights(curr_eval_model.get_weights())
    baseline_eval_model.compile()

    baseline_action_model = tf.keras.models.clone_model(curr_action_model)
    baseline_action_model.set_weights(curr_action_model.get_weights())
    baseline_action_model.compile()

    for layer in baseline_bet_model.layers:
        layer.trainable = False
    for layer in baseline_eval_model.layers:
        layer.trainable = False
    for layer in baseline_action_model.layers:
        layer.trainable = False

    print("Initialized baseline models as untrainable copies of current models.")

    return curr_bet_model, baseline_bet_model, curr_eval_model, baseline_eval_model, curr_action_model, baseline_action_model

def saveModels(run_name, bet_model, eval_model, act_model):
    bet_model_path = os.path.join(os.getcwd(),run_name,"best_bet_model")
    bet_model.save(bet_model_path)

    eval_model_path = os.path.join(os.getcwd(),run_name,"best_eval_model")
    eval_model.save(eval_model_path)

    act_model_path = os.path.join(os.getcwd(),run_name,"best_act_model")
    act_model.save(act_model_path)
    print("Saved bet, eval, and action models.")

def playJudgmentGameThread(core_id, curr_action_weights, curr_bet_weights, curr_eval_weights, epsilon_choice):
    """
    Plays a game of Judgment with an asynchronous actor, given models and a randomly chosen epsilon value.

    At the end of each round of the game, the actor will compute a gradient update of its current weights.
    Along the way, it accumulates the gradients for the action, bet, and eval models, eventually applying them
    to the global weight network.
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

    for game_num in range(nn_config.A3C_NUM_GAMES_PER_WORKER):
        print(f"Starting game {game_num+1}/{nn_config.A3C_NUM_GAMES_PER_WORKER} on core {core_id}",end="\r")
        #Initialize game
        jg = JudgmentGame([DQNAgent(0,epsilon_choice,load_models=False), DQNAgent(1,epsilon_choice,load_models=False), \
                           DQNAgent(2,epsilon_choice,load_models=False), DQNAgent(3,epsilon_choice,load_models=False)])
        
        #Set models for each agent
        for agent in jg.agents:
            agent.action_model = thread_action_model
            agent.bet_model = thread_bet_model
            agent.eval_model = thread_eval_model

        start = time.time()
        bet_train_data, eval_train_data, action_train_data = jg.playGameAndCollectData()
        print(f"Game {game_num+1}/{nn_config.A3C_NUM_GAMES_PER_WORKER} on core {core_id} took {time.time()-start} seconds.")

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
            #NOTE: there is the potential for a huge speedup here: playGameAndCollectData()
            #returns data in the form of (bet_state, reward), at which point we reevaluate the bet network
            #on that bet state in order to compute the loss. If we made a new function which returned simply
            #the Q value and the loss, that would be far quicker.
            
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

    return accum_act_gradients, accum_bet_gradients, accum_eval_gradients


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

    curr_bet_model, baseline_bet_model, curr_eval_model, baseline_eval_model, curr_action_model, baseline_action_model = loadModels(args)

    performance_against_best_agents = []
    new_states_to_achieve_performances = []

    new_state_action_pairs_trained_on = 0

    best_bet_weights = copy(curr_bet_model.get_weights())
    best_action_weights = copy(curr_action_model.get_weights())
    best_eval_weights = copy(curr_eval_model.get_weights())

    epsilon_choices = [0.3, 0.4, 0.2]
    num_global_updates = 0

    best_model_beat_baseline_by = 0
    iterations_without_improving_best_agent = 0

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
        with Pool(processes=nn_config.A3C_NUM_WORKERS) as p:
            worker_accum_gradients = p.starmap(playJudgmentGameThread, game_arguments)

        #Accumulate gradients from each worker
        for (worker_accum_act_gradients, worker_accum_bet_gradients, worker_accum_eval_gradients) in worker_accum_gradients:
            act_gradients = act_gradients + worker_accum_act_gradients
            bet_gradients = bet_gradients + worker_accum_bet_gradients
            eval_gradients = eval_gradients + worker_accum_eval_gradients

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=nn_config.LEARNING_RATE)

        #update weights by the accumulated gradients
        optimizer.apply_gradients(zip(act_gradients, curr_action_model.trainable_variables))
        optimizer.apply_gradients(zip(bet_gradients, curr_bet_model.trainable_variables))
        optimizer.apply_gradients(zip(eval_gradients, curr_eval_model.trainable_variables))

        num_global_updates += 1

        print(f"Finished global update {num_global_updates}!")

        if num_global_updates % nn_config.A3C_GLOBAL_NET_UPDATE_FREQ == 0:
            #~~~~~~~~~~~~~~~~~~~~~~EVALUATING PERFORMANCE~~~~~~~~~~~~~~~~~~~~~~~
            print(f"{nn_config.A3C_GLOBAL_NET_UPDATE_FREQ} global updates have occured, so evaluating performance.")

            #epsilon is zero for evaluation. Load new models into agents 0 and 1, best models into agents 2 and 3.
            agents_to_compare = [DQNAgent(0,load_models=False),DQNAgent(1,load_models=False),DQNAgent(2,load_models=False),DQNAgent(3,load_models=False)]
            for agent in agents_to_compare[0:2]:
                agent.action_model = curr_action_model
                agent.bet_model = curr_bet_model
                agent.eval_model = curr_eval_model

                agent.action_model.compile()
                agent.bet_model.compile()
                agent.eval_model.compile()

            for agent in agents_to_compare[2:]:
                agent.action_model = baseline_action_model
                agent.bet_model = baseline_bet_model
                agent.eval_model = baseline_eval_model

                agent.action_model.compile()
                agent.bet_model.compile()
                agent.eval_model.compile()

            print("Performance against baseline agent:")
            avg_scores_against_baseline_agents = compareAgents(agents_to_compare,games_num=24, cores=cpu_count())
            new_agent_score_against_baseline = sum(avg_scores_against_baseline_agents[0:2])/len(avg_scores_against_baseline_agents[0:2])
            baseline_agent_score_against_new = sum(avg_scores_against_baseline_agents[2:])/len(avg_scores_against_baseline_agents[2:])
            print(f"New Agent average score: {new_agent_score_against_baseline}, baseline agent average score: {baseline_agent_score_against_new}")

            current_beat_baseline_by = new_agent_score_against_baseline - baseline_agent_score_against_new

            if (current_beat_baseline_by > best_model_beat_baseline_by):
                print(f"!!!New agent improves on current best agent (beating baseline by {current_beat_baseline_by} instead of {best_model_beat_baseline_by}), so saving it!!!")
                best_bet_weights = copy(curr_bet_model.get_weights())
                best_action_weights = copy(curr_action_model.get_weights())
                best_eval_weights = copy(curr_eval_model.get_weights())

                saveModels(args.run_name, curr_bet_model, curr_eval_model, curr_action_model)

                iterations_without_improving_best_agent = 0
                best_model_beat_baseline_by = current_beat_baseline_by
            else:
                iterations_without_improving_best_agent += 1
                
                if iterations_without_improving_best_agent >= 3:
                    print(f"It has been {iterations_without_improving_best_agent} iterations without improving on best agent, so reset to old best agent.")
                    #remove the last ~50 games of data from the buffer
                    action_rb_data = [action_rb_data.pop() for _i in range(3*nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN)]
                    eval_rb_data = [eval_rb_data.pop() for _i in range(3*nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN//3)] 
                    bet_rb_data = [bet_rb_data.pop() for _i in range(3*nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN//6)]

                    curr_bet_model.set_weights(best_bet_weights)
                    curr_action_model.set_weights(best_action_weights)
                    curr_eval_model.set_weights(best_eval_weights)

                    iterations_without_improving_best_agent = 0
                else: print(f"~~~New agent does not improve on best agent (beat baseline by {current_beat_baseline_by} instead of {best_model_beat_baseline_by}), so increase iterations without improving on best to {iterations_without_improving_best_agent} and continue training.~~~")

            if args.track: wandb.log({"eval/score_diff_against_baseline": current_beat_baseline_by, "eval/state_transitions": new_state_action_pairs_trained_on})

            #save data for progress plots
            new_states_to_achieve_performances.append(new_state_action_pairs_trained_on)
            performance_against_best_agents.append(current_beat_baseline_by)
            plt.plot(new_states_to_achieve_performances,performance_against_best_agents)
            plt.xlabel("Number of new training examples")
            baseline_model_name = args.action_model_path.split("/")[0]
            plt.ylabel(f"Avg. Score Diff vs. Baseline Model ({baseline_model_name})")

            progress_graph_path = os.path.join(os.getcwd(), args.run_name, "performance_vs_baseline_agent_over_time.png")
            plt.savefig(progress_graph_path)
            plt.clf()

if __name__ == "__main__":
    # jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])
    # bet_exp_data, eval_exp_data, state_transition_bank = loadExperienceData("run1")

    trainAgentViaA3C()

    # bet_data, eval_data, state_transitions = jg.playGameAndTrackStateTransitions()

    # for state_transition in state_transitions:
    #     start_srs = state_transition[0]
    #     action = state_transition[1]
    #     final_srs = state_transition[2]

    #     order_position = len(start_srs.card_stack)
    #     print(order_position)
    #     curr_agent = start_srs.agents[order_position]
    #     print(f"Start hand:")
    #     for card in curr_agent.hand:
    #         print(card.name)

    #     print(f"Playing card: {action.name}")
    #     if type(final_srs) != type(0.0):
    #         order_position = len(final_srs.card_stack)
    #         curr_agent = final_srs.agents[order_position]
    #         print(f"Final hand:")
    #         for card in curr_agent.hand:
    #             print(card.name)
    #     else:
    #         print(f"Round over. Reward = {final_srs}")

    #     print("~~~~~~~~~")

    # for bet_dat in bet_data:
    #     print(bet_dat[0][59],bet_dat[1])