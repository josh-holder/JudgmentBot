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

from JudgmentUtils import postProcessTrainData, postProcessBetTrainData, convertSubroundSituationToActionState
from JudgmentGame import JudgmentGame
from DQNAgent import DQNAgent
from agent_compare import compareAgents
from HumanBetAgent import HumanBetAgent

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

def loadExperienceData(run_name):
    """
    Loads exists, or creates new experience data to use for experience replay.
    """
    folder_name = "dqn_experience_data"
    run_folder_path = os.path.join(os.getcwd(),run_name,folder_name)
    
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)

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
    
    act_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"act_experience_data.pkl")
    if os.path.exists(act_mem_path):
        print("Loading existing action experience data.")
        with open(act_mem_path,'rb') as f:
            act_experience_data = pickle.load(f)
        print(f"Done loading {len(act_experience_data)} items of action experience data")
    else:
        print("Previous action experience data not found: generating empty memory list.")
        act_experience_data = deque(maxlen=ACTION_EXPERIENCE_BANK_SIZE)

    return bet_experience_data, eval_experience_data, act_experience_data

def saveExperienceData(run_name,bet_exp_data,eval_exp_data,state_transition_bank):
    folder_name = "dqn_experience_data"

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
    [(init_srs, card, final_srs),...,(init_srs, card, final_srs)]

    to the form:

    [([action_state],reward),...,([action_state],reward)]

    (Later, we can use postProcessTrainData() to convert this to training form)
    """
    converted_minibatch = []
    for state_transition in state_transition_minibatch:
        srs_before = state_transition[0]
        card = state_transition[1]
        srs_after = state_transition[2]

        #Determine current agent, set evaluation models
        curr_agent = srs_before.agents[len(srs_before.card_stack)]
        curr_agent.eval_model = eval_model

        action_state = convertSubroundSituationToActionState(state_transition[0],curr_agent,card)

        #If srs_after is a float, this is the reward.
        if type(srs_after) == type(1.0): reward = srs_after
        #Otherwise, the reward is the maximum Q-value of the next state.
        else:
            next_agent = srs_after.agents[len(srs_after.card_stack)]
            next_agent.action_model = action_model
            next_agent.eval_model = eval_model

            next_agent.determineCardOptions(srs_after)

            best_act_val = -np.inf
            for available_card in next_agent.available_cards:
                next_action_state = convertSubroundSituationToActionState(srs_after,next_agent,available_card)

                next_action_state = [act_state_component[np.newaxis,:] for act_state_component in next_action_state]
                act_val = float(next_agent.action_model(next_action_state).numpy()[0]) #potentially use squeeze method from stackoverflow

                if act_val > best_act_val:
                    best_act_val = act_val
            reward = best_act_val

        converted_minibatch.append((action_state, reward))

    return converted_minibatch
            

def trainDQNAgent():
    parser = _build_parser()
    args = parser.parse_args()

    bet_exp_data, eval_exp_data, state_transition_bank = loadExperienceData(args.run_name)

    jg = JudgmentGame(agents=[DQNAgent(0,epsilon=0.15, load_models=False),DQNAgent(1,epsilon=0.15, load_models=False),DQNAgent(2,epsilon=0.15, load_models=False),DQNAgent(3,epsilon=0.15, load_models=False)])

    print("Loading models...")
    bet_model_path = os.path.join(os.getcwd(),args.bet_model_path)
    bet_model = keras.models.load_model(bet_model_path)

    eval_model_path = os.path.join(os.getcwd(),args.eval_model_path)
    eval_model = keras.models.load_model(eval_model_path)

    action_model_path = os.path.join(os.getcwd(),args.action_model_path)
    action_model = keras.models.load_model(action_model_path)

    for agent in jg.agents:
        agent.action_model = action_model
        agent.bet_model = bet_model
        agent.eval_model = eval_model

    #Wait until all experience banks are at least 1/4 full to start learning:
    # while len(state_transition_bank) < 250:
    print("Generating initial amount of training data...")
    while len(bet_exp_data)<BET_EXPERIENCE_BANK_SIZE/4 or len(eval_exp_data)<EVAL_EXPERIENCE_BANK_SIZE/4 \
        or len(state_transition_bank)<ACTION_EXPERIENCE_BANK_SIZE/4:
        bet_data, eval_data, state_transitions = jg.playGameAndTrackStateTransitions()

        #add to existing bet_exp_data bank
        bet_exp_data.extend(bet_data)
        eval_exp_data.extend(eval_data)
        state_transition_bank.extend(state_transitions)

        print(f"Bet: {len(bet_exp_data)}/{BET_EXPERIENCE_BANK_SIZE/4}, Eval: {len(eval_exp_data)}/{EVAL_EXPERIENCE_BANK_SIZE/4}, Act: {len(state_transition_bank)}/{ACTION_EXPERIENCE_BANK_SIZE/4}",end='\r')

        jg.resetGame()

    print(f"Sufficient training data is available ({len(state_transition_bank)} state transition, {len(eval_exp_data)} eval, {len(bet_exp_data)} bet)")

    performance_against_humanbet = [25] #conservative estimate for how much the base expert-trained agent beats HumanBet by
    times_to_achieve_performances = [0]

    training_start = time.time()

    beat_humanbet_by = 0 #track how much the previous agent beat the HumanBetAgent by to determine if the nnew agent is better

    while True:
        new_state_transitions = 0

        old_action_model = action_model
        old_bet_model = bet_model
        old_eval_model = eval_model

        num_new_transitions_before_eval_bet_training = 25000
        print(f"Playing games and training action model for {num_new_transitions_before_eval_bet_training} state transitions")
        while new_state_transitions < num_new_transitions_before_eval_bet_training:
            bet_data, eval_data, state_transitions = jg.playGameAndTrackStateTransitions()

            new_state_transitions += len(state_transitions)
            print(f"State transitions: {new_state_transitions}/{num_new_transitions_before_eval_bet_training}",end='\r')

            #add to existing bet_exp_data bank
            bet_exp_data.extend(bet_data)
            eval_exp_data.extend(eval_data)
            state_transition_bank.extend(state_transitions)

            minibatch = random.sample(state_transition_bank,min(RETRAIN_BATCH_SIZE, len(state_transition_bank))) #selects BATCH_SIZE unique states to retrain the NN on

            #Convert minibatch from (init_srs, action, final_srs) to (action_state, reward) with reward calculated with current action_model
            minibatch = convertMiniBatchToStateRewardPair(minibatch, action_model, eval_model)
            minibatch_inputs = []
            minibatch_outputs = []

            #Convert minibatch to correct form for inputs and outputs to train model on
            for minibatch_input_output in minibatch:
                minibatch_inputs.append(minibatch_input_output[0])
                minibatch_outputs.append(minibatch_input_output[1])
            minibatch_inputs = postProcessTrainData(minibatch_inputs)
            minibatch_outputs = np.array(minibatch_outputs)

            fit = action_model.fit(minibatch_inputs, minibatch_outputs, epochs=RETRAIN_EPOCHS, batch_size=32, verbose = 0)

            jg.resetGame()

            #Set action models of agents to the new action model
            for agent in jg.agents:
                agent.action_model = action_model

        saveExperienceData(args.run_name, bet_exp_data, eval_exp_data, state_transition_bank)

        #~~~~~~~~~~~~~~~~~~~~~~~TRAINING BET AND EVAL NETWORKS ON NEW EXPERIENCE DATA~~~~~~~~~~~~~~~~~~~``
        print(f">{num_new_transitions_before_eval_bet_training} new transitions generated, so retraining bet and evaluation networks on new data.")

        bet_data_inputs = []
        bet_data_outputs = []
        for bet_data in bet_exp_data:
            bet_data_inputs.append(bet_data[0])
            bet_data_outputs.append(bet_data[1])

        bet_data_inputs = postProcessBetTrainData(bet_data_inputs)
        bet_data_outputs = np.array(bet_data_outputs)
        
        print("Retraining bet network...")
        bet_model.fit(bet_data_inputs,bet_data_outputs,epochs=128,batch_size=256,verbose=0)
        print("Done retraining bet network.")

        eval_data_inputs = []
        eval_data_outputs = []
        for eval_data in eval_exp_data:
            eval_data_inputs.append(eval_data[0])
            eval_data_outputs.append(int(eval_data[1])) #TODO: convert boolean outputs to int outputs

        eval_data_inputs = postProcessTrainData(eval_data_inputs)
        eval_data_outputs = np.array(eval_data_outputs)

        print("Retraining eval network...")
        eval_model.fit(eval_data_inputs,eval_data_outputs,epochs=128,batch_size=256,verbose=0)
        print("Done retraining eval network.")

        #Updating action and evaluation models for agents
        for agent in jg.agents:
            agent.bet_model = bet_model
            agent.eval_model = eval_model

        #~~~~~~~~~~~~~~~~~~~~~~EVALUATING PERFORMANCE~~~~~~~~~~~~~~~~~~~~~~~
        print("Bet and Evaluation models retrained on new data: evaluating performance.")

        agents_to_compare = [DQNAgent(0,load_models=False),DQNAgent(1,load_models=False),DQNAgent(2,load_models=False),DQNAgent(3,load_models=False)]
        agents_to_compare[0].action_model = action_model
        agents_to_compare[0].bet_model = bet_model
        agents_to_compare[0].eval_model = eval_model

        for agent in agents_to_compare[1:]:
            agent.action_model = old_action_model
            agent.bet_model = old_bet_model
            agent.eval_model = old_eval_model

        print("Performance against previous agent iteration:")
        avg_scores_against_prev_agents = compareAgents(agents_to_compare,games_num=5)
        new_agent_score_against_prev = avg_scores_against_prev_agents[0]
        prev_agent_score_against_new = sum(avg_scores_against_prev_agents[1:])/len(avg_scores_against_prev_agents[1:])
        print(f"New Agent average score: {new_agent_score_against_prev}, previous agent average score: {prev_agent_score_against_new}")

        print("Performance against HumanBetAgents:")
        avg_scores_against_humanbet_agents = compareAgents([agents_to_compare[0],HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)],games_num=5)
        new_agent_score_against_humanbet = avg_scores_against_humanbet_agents[0]
        humanbet_agent_score_against_new = sum(avg_scores_against_humanbet_agents[1:])/len(avg_scores_against_humanbet_agents[1:])
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
        times_to_achieve_performances.append((time.time()-training_start)/60)
        plt.plot(times_to_achieve_performances,performance_against_humanbet)
        plt.xlabel("Time (min)")
        plt.ylabel("Difference in average score between new agent and HumanBet agent")

        progress_graph_path = os.path.join(os.getcwd(), args.run_name, "performance_over_time.png")
        plt.savefig(progress_graph_path)

if __name__ == "__main__":
    # jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])

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