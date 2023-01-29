from collections import deque
import pickle
import os
from nn_config import ACTION_EXPERIENCE_BANK_SIZE, BET_EXPERIENCE_BANK_SIZE, EVAL_EXPERIENCE_BANK_SIZE, \
    RETRAIN_BATCH_SIZE, RETRAIN_EPOCHS
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

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def _build_parser():
    parser = argparse.ArgumentParser(description='Run NN to generate ideal playing strategy for SPLT.')

    parser.add_argument(
        '-r','--run_name',
        help="Will name the output directory for the results of the run",
        type=str,
        default="run1",
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

    return parser

def loadExperienceData(run_name, folder_name='dqn_experience_data'):
    """
    Loads exists, or creates new experience data to use for experience replay.
    """
    run_folder_path = os.path.join(os.getcwd(),run_name)
    dqnexp_folder_path = os.path.join(run_folder_path,folder_name)
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)
    if not os.path.exists(dqnexp_folder_path):
        os.mkdir(dqnexp_folder_path)

    act_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"act_experience_data.pkl")
    if os.path.exists(act_mem_path):
        print("Loading existing action experience data.")
        with open(act_mem_path,'rb') as f:
            act_experience_data = pickle.load(f)
        print(f"Done loading {len(act_experience_data)} items of action experience data")
    else:
        print("Previous action experience data not found: generating empty memory list.")
        act_experience_data = deque(maxlen=ACTION_EXPERIENCE_BANK_SIZE)

    bet_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"bet_experience_data.pkl")
    if os.path.exists(bet_mem_path):
        print("Loading existing bet experience data.")
        with open(bet_mem_path,'rb') as f:
            bet_experience_data = pickle.load(f)
        print(f"Done loading {len(bet_experience_data)} items of bet experience data")
    else:
        print("Previous bet experience data not found: generating empty memory list.")
        bet_experience_data = deque(maxlen=BET_EXPERIENCE_BANK_SIZE)

    eval_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"eval_experience_data.pkl")
    if os.path.exists(eval_mem_path):
        print("Loading existing eval experience data.")
        with open(eval_mem_path,'rb') as f:
            eval_experience_data = pickle.load(f)
        print(f"Done loading {len(eval_experience_data)} items of eval experience data")
    else:
        print("Previous eval experience data not found: generating empty memory list.")
        eval_experience_data = deque(maxlen=EVAL_EXPERIENCE_BANK_SIZE)

    return bet_experience_data, eval_experience_data, act_experience_data

def saveExperienceData(run_name,bet_exp_data,eval_exp_data,state_transition_bank,folder_name="dqn_experience_data"):
    folder_path = os.path.join(os.getcwd(), run_name, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    bet_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"bet_experience_data.pkl")
    with open(bet_mem_path,'wb') as f:
        print(f"Saving {len(bet_exp_data)} items of bet experience data in {bet_mem_path}")
        pickle.dump(bet_exp_data,f)
    
    eval_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"eval_experience_data.pkl")
    with open(eval_mem_path,'wb') as f:
        print(f"Saving {len(eval_exp_data)} items of eval experience data in {eval_mem_path}")
        pickle.dump(eval_exp_data,f)
    
    act_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"act_experience_data.pkl")
    with open(act_mem_path,'wb') as f:
        print(f"Saving {len(state_transition_bank)} items of state transition data in {act_mem_path}")
        pickle.dump(state_transition_bank,f)

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
            

def trainAgentViaA3C():
    parser = _build_parser()
    args = parser.parse_args()

    run_folder_path = os.path.join(os.getcwd(),args.run_name)
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)

    print("Loading models...")
    bet_model_path = os.path.join(os.getcwd(),args.bet_model_path)
    bet_model = keras.models.load_model(bet_model_path)
    print(f"Loaded bet model from {bet_model_path}")

    eval_model_path = os.path.join(os.getcwd(),args.eval_model_path)
    eval_model = keras.models.load_model(eval_model_path)
    print(f"Loaded eval model from {eval_model_path}")

    action_model_path = os.path.join(os.getcwd(),args.action_model_path)
    action_model = keras.models.load_model(action_model_path)
    print(f"Loaded action model fron {action_model_path}")

    for agent in jg.agents:
        agent.action_model = action_model
        agent.bet_model = bet_model
        agent.eval_model = eval_model

    performance_against_humanbet = []
    performance_against_prev_agents = []
    new_states_to_achieve_performances = []

    new_state_action_pairs_trained_on = 0
    beat_humanbet_by = 0 #track how much the previous agent beat the HumanBetAgent by to determine if the nnew agent is better

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

        act_gradients = Queue()
        act_gradients.put([zero gradients])
        #bet and eval gradients
        for core_num in range(cpu_count()):
            epsilon_choice = random.choice(epsilon_choices)
            p = Process(target=playJudgmentGameThread(action_model, bet_model, eval_model, epsilon_choice)) 

            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        #update weights by the accumulated gradients

        #evaluate performance every 100 games maybe
        #~~~~~~~~~~~~~~~~~~~~~~EVALUATING PERFORMANCE~~~~~~~~~~~~~~~~~~~~~~~
        print("Bet and Evaluation models retrained on new data: evaluating performance.")

        #epsilon is zero for evaluation. Load new models into agents 0 and 1, old models into agents 2 and 3.
        agents_to_compare = [DQNAgent(0,load_models=False),DQNAgent(1,load_models=False),DQNAgent(2,load_models=False),DQNAgent(3,load_models=False)]
        for agent in agents_to_compare[0:2]:
            agent.action_model = action_model
            agent.bet_model = bet_model
            agent.eval_model = eval_model

        for agent in agents_to_compare[2:]:
            agent.action_model = old_action_model
            agent.bet_model = old_bet_model
            agent.eval_model = old_eval_model

        print("Performance against previous agent iteration:")
        avg_scores_against_prev_agents = compareAgents(agents_to_compare,games_num=20, cores=cpu_count())
        new_agent_score_against_prev = sum(avg_scores_against_prev_agents[0:2])/len(avg_scores_against_prev_agents[0:2])
        prev_agent_score_against_new = sum(avg_scores_against_prev_agents[2:])/len(avg_scores_against_prev_agents[2:])
        print(f"New Agent average score: {new_agent_score_against_prev}, previous agent average score: {prev_agent_score_against_new}")

        print("Performance against HumanBetAgents:")
        avg_scores_against_humanbet_agents = compareAgents([agents_to_compare[0],agents_to_compare[1],HumanBetAgent(2),HumanBetAgent(3)],games_num=20, cores=cpu_count())
        new_agent_score_against_humanbet = sum(avg_scores_against_humanbet_agents[0:2])/len(avg_scores_against_humanbet_agents[0:2])
        humanbet_agent_score_against_new = sum(avg_scores_against_humanbet_agents[2:])/len(avg_scores_against_humanbet_agents[2:])
        print(f"New Agent average score: {new_agent_score_against_humanbet}, HumanBet agent average score: {humanbet_agent_score_against_new}")

        if (new_agent_score_against_prev > prev_agent_score_against_new) or (new_agent_score_against_humanbet > beat_humanbet_by):
            print("New agent improves on old agent, so saving it.")
            saveModels(args.run_name, bet_model, eval_model, action_model)
            beat_humanbet_by = new_agent_score_against_humanbet - humanbet_agent_score_against_new
        else:
            print("New agent does not improve on old agent, so does not save the model.")
            bet_model = old_bet_model
            eval_model = old_eval_model
            action_model = old_action_model

        print(f"Evaluation complete: generating {num_new_transitions_before_eval_bet_training} new state transitions.")

        #save data for progress plots
        performance_against_humanbet.append(new_agent_score_against_humanbet-humanbet_agent_score_against_new)
        new_states_to_achieve_performances.append(new_state_action_pairs_trained_on)
        plt.plot(new_states_to_achieve_performances,performance_against_humanbet, label='vs. HumanBet')
        plt.xlabel("Number of new training examples")
        plt.ylabel("Avg. Score Difference vs. HumanBet")

        performance_against_prev_agents.append(new_agent_score_against_prev-prev_agent_score_against_new)
        plt.plot(new_states_to_achieve_performances,performance_against_prev_agents, label='vs. previous iteration')
        plt.xlabel("Number of new training examples")
        plt.ylabel("Avg. Score Diff vs. Prev. Iteration")

        progress_graph_path = os.path.join(os.getcwd(), args.run_name, "performance_vs_prev_agent_over_time.png")
        plt.legend()
        plt.savefig(progress_graph_path)
        plt.clf()

if __name__ == "__main__":
    # jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])
    # bet_exp_data, eval_exp_data, state_transition_bank = loadExperienceData("run1")

    trainDQNAgent()

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