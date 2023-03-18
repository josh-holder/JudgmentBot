from collections import deque
import pickle
import os
import nn_config
import argparse
import numpy as np
import random
import keras
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from copy import deepcopy
from JudgmentUtils import postProcessTrainData, postProcessBetTrainData, convertSubroundSituationToActionState
from JudgmentGame import JudgmentGame
from DQNAgent import DQNAgent
from compare_agents import compareAgents
from HumanBetAgent import HumanBetAgent
from multiprocessing import cpu_count, Queue
import wandb
from copy import copy, deepcopy
from multiprocessing import Process, cpu_count, Queue

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

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
        "--wandb-entity", 
        type=str, 
        default="josh-holder", 
        help="the entity (team) of wandb's project"
    )

    return parser

def initWandBTrack(args):
    #initialize weights and biases tracking
    run_name = f"{args.run_name}__{int(time.time())}"
    config_dict = vars(args)
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
    if args.models_path == None:
        curr_bet_model_path = os.path.join(os.getcwd(),args.bet_model_path)
        curr_bet_model = keras.models.load_model(curr_bet_model_path)
        print(f"Loaded current bet model from {curr_bet_model_path}")

        curr_eval_model_path = os.path.join(os.getcwd(),args.eval_model_path)
        curr_eval_model = keras.models.load_model(curr_eval_model_path)
        print(f"Loaded current eval model from {curr_eval_model_path}")

        curr_action_model_path = os.path.join(os.getcwd(),args.action_model_path)
        curr_action_model = keras.models.load_model(curr_action_model_path)
        print(f"Loaded current action model fron {curr_action_model_path}")
    
    else:
        curr_bet_model_path = os.path.join(os.getcwd(),args.models_path,"best_bet_model")
        curr_bet_model = keras.models.load_model(curr_bet_model_path)
        print(f"Loaded current bet model from {curr_bet_model_path}")

        curr_eval_model_path = os.path.join(os.getcwd(),args.models_path,"best_eval_model")
        curr_eval_model = keras.models.load_model(curr_eval_model_path)
        print(f"Loaded current eval model from {curr_eval_model_path}")

        curr_action_model_path = os.path.join(os.getcwd(),args.models_path,"best_act_model")
        curr_action_model = keras.models.load_model(curr_action_model_path)
        print(f"Loaded current action model fron {curr_action_model_path}")

    baseline_bet_model = tf.keras.models.clone_model(curr_bet_model)
    baseline_bet_model.set_weights(curr_bet_model.get_weights())

    baseline_eval_model = tf.keras.models.clone_model(curr_eval_model)
    baseline_eval_model.set_weights(curr_eval_model.get_weights())

    baseline_action_model = tf.keras.models.clone_model(curr_action_model)
    baseline_action_model.set_weights(curr_action_model.get_weights())

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

def convertMiniBatchToStateRewardPair(state_transition_minibatch, action_model, eval_model):
    """
    Convert minibatch of state transitions of the form 
    [(init_srs, card, reward),...,(init_srs, card, reward)]

    to the form:

    [([action_state],reward),...,([action_state],reward)]

    (Later, we can use postProcessTrainData() to convert this to training form)

    (NOTE: 1/27/23 changed to use rewards directly instead of transitions for lower bias from using direct Monte Carlo rewards,
    lower storage needs, lower computation requirements, etc. Form used to be:
    [(init_srs, card, final_srs),...,(init_srs, card, final_srs)])
    """
    converted_minibatch = []
    for state_transition in state_transition_minibatch:
        srs_before = state_transition[0]
        card = state_transition[1]
        reward = state_transition[2]

        #Determine current agent, set evaluation models
        curr_agent = deepcopy(srs_before.agents[len(srs_before.card_stack)])
        curr_agent.eval_model = eval_model

        action_state = convertSubroundSituationToActionState(srs_before,curr_agent,card)

        converted_minibatch.append((action_state, reward))

    return converted_minibatch

def playJudgmentGameThread(curr_action_model, curr_bet_model, curr_eval_model, epsilon_choice, \
                           accum_act_gradients_queue, accum_bet_gradients_queue, accum_eval_gradients_queue):
    """
    Plays a game of Judgment with an asynchronous actor, given models and a randomly chosen epsilon value.

    At the end of each round of the game, the actor will compute a gradient update of its current weights.
    Along the way, it accumulates the gradients for the action, bet, and eval models, eventually applying them
    to the global weight network.
    """

    #Initialize local models in thread
    thread_action_model = tf.keras.models.clone_model(curr_action_model)
    thread_action_model.set_weights(curr_action_model.get_weights())

    thread_bet_model = tf.keras.models.clone_model(curr_bet_model)
    thread_bet_model.set_weights(curr_bet_model.get_weights())

    thread_eval_model = tf.keras.models.clone_model(curr_eval_model)
    thread_eval_model.set_weights(curr_eval_model.get_weights())

    for game_num in range(nn_config.A3C_NUM_GAMES_PER_WORKER):
        #Initialize game
        jg = JudgmentGame([DQNAgent(0,epsilon_choice,load_models=False), DQNAgent(1,epsilon_choice,load_models=False), \
                           DQNAgent(2,epsilon_choice,load_models=False), DQNAgent(3,epsilon_choice,load_models=False)])
        
        #Set models for each agent
        for agent in jg.agents:
            agent.action_model = thread_action_model
            agent.bet_model = thread_bet_model
            agent.eval_model = thread_eval_model

        bet_train_data, eval_train_data, action_train_data = jg.playGameAndCollectData()
        print(type(bet_train_data))

        #Compute loss on bet model
        print(len(bet_train_data))
        with tf.GradientTape() as bet_tape:
            #NOTE: there is the potential for a huge speedup here: playGameAndCollectData()
            #returns data in the form of (bet_state, reward), at which point we reevaluate the bet network
            #on that bet state in order to compute the loss. If we made a new function which returned simply
            #the Q value and the loss, that would be far quicker.
            for bet_train_datum in bet_train_data:
                bet_state = bet_train_datum[0]
                print(bet_state)
                prediction = thread_bet_model(bet_state)
                reward = bet_train_datum[1]
                bet_loss = tf.keras.losses.MeanSquaredError(reward, prediction)
                print(bet_loss)


def trainAgentViaA3C():
    parser = _build_parser()
    args = parser.parse_args()

    run_folder_path = os.path.join(os.getcwd(),args.run_name)
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)

    curr_bet_model, baseline_bet_model, curr_eval_model, baseline_eval_model, curr_action_model, baseline_action_model = loadModels(args)

    performance_against_best_agents = []
    new_states_to_achieve_performances = []

    new_state_action_pairs_trained_on = 0

    """
    Psuedocode:
    -load initial models
    - while True:
        -initialize 4 actors, each with the model, and with a randomly chosen epsilon
        -each play a game, accumulate gradients for action, bet, and eval networks, as well as how many examples they're from
        -perform a gradient update on the models
        -repeat

    This removes need to store replay buffer
    """
    epsilon_choices = [0.3, 0.4, 0.2]
    while True:
        processes = []

        accum_act_gradients_queue = Queue()
        accum_act_gradients = [tf.zeros_like(var) for var in curr_action_model.trainable_variables]
        accum_act_gradients_queue.put(accum_act_gradients)

        accum_bet_gradients_queue = Queue()
        accum_bet_gradients = [tf.zeros_like(var) for var in curr_bet_model.trainable_variables]
        accum_bet_gradients_queue.put(accum_bet_gradients)

        accum_eval_gradients_queue = Queue()
        accum_eval_gradients = [tf.zeros_like(var) for var in curr_eval_model.trainable_variables]
        accum_eval_gradients_queue.put(accum_eval_gradients)

        #bet and eval gradients
        for _ in range(1):
            epsilon_choice = random.choice(epsilon_choices)
            p = Process(target=playJudgmentGameThread(curr_action_model, curr_bet_model, curr_eval_model, epsilon_choice, \
                                                      accum_act_gradients_queue, accum_bet_gradients_queue, accum_eval_gradients_queue)) 

            processes.append(p)
            p.start()

        for p in processes:
            print("joining")
            p.join()
            print("joined")
        
        print("done")
        break

        #update weights by the accumulated gradients

        #evaluate performance every 100 games maybe
        #~~~~~~~~~~~~~~~~~~~~~~EVALUATING PERFORMANCE~~~~~~~~~~~~~~~~~~~~~~~
        print("Bet and Evaluation models retrained on new data: evaluating performance.")

        #epsilon is zero for evaluation. Load new models into agents 0 and 1, best models into agents 2 and 3.
        agents_to_compare = [DQNAgent(0,load_models=False),DQNAgent(1,load_models=False),DQNAgent(2,load_models=False),DQNAgent(3,load_models=False)]
        for agent in agents_to_compare[0:2]:
            agent.action_model = curr_action_model
            agent.bet_model = curr_bet_model
            agent.eval_model = curr_eval_model

        for agent in agents_to_compare[2:]:
            agent.action_model = baseline_action_model
            agent.bet_model = baseline_bet_model
            agent.eval_model = baseline_eval_model

        print("Performance against baseline agent:")
        avg_scores_against_baseline_agents = compareAgents(agents_to_compare,games_num=24, cores=cpu_count())
        new_agent_score_against_baseline = sum(avg_scores_against_baseline_agents[0:2])/len(avg_scores_against_baseline_agents[0:2])
        baseline_agent_score_against_new = sum(avg_scores_against_baseline_agents[2:])/len(avg_scores_against_baseline_agents[2:])
        print(f"New Agent average score: {new_agent_score_against_baseline}, baseline agent average score: {baseline_agent_score_against_new}")

        current_beat_baseline_by = new_agent_score_against_baseline - baseline_agent_score_against_new

        # print("Performance against HumanBetAgents:")
        # avg_scores_against_humanbet_agents = compareAgents([agents_to_compare[0],agents_to_compare[1],HumanBetAgent(2),HumanBetAgent(3)],games_num=20, cores=cpu_count())
        # new_agent_score_against_humanbet = sum(avg_scores_against_humanbet_agents[0:2])/len(avg_scores_against_humanbet_agents[0:2])
        # humanbet_agent_score_against_new = sum(avg_scores_against_humanbet_agents[2:])/len(avg_scores_against_humanbet_agents[2:])
        # print(f"New Agent average score: {new_agent_score_against_humanbet}, HumanBet agent average score: {humanbet_agent_score_against_new}")

        if (current_beat_baseline_by > best_model_beat_baseline_by):
            print(f"!!!New agent improves on current best agent (beating baseline by {current_beat_baseline_by} instead of {best_model_beat_baseline_by}), so saving it.!!!")
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

        print(f"Evaluation complete: generating {num_new_transitions_before_eval_bet_training} new state transitions.")

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
    print("really done")

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