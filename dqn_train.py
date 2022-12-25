from collections import deque
import pickle
import os
from nn_config import ACTION_EXPERIENCE_BANK_SIZE, BET_EXPERIENCE_BANK_SIZE, EVAL_EXPERIENCE_BANK_SIZE
import argparse
import numpy as np

from JudgmentUtils import postProcessTrainData, postProcessBetTrainData
from JudgmentGame import JudgmentGame
from DQNAgent import DQNAgent

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def _build_parser():
    parser = argparse.ArgumentParser(description='Run NN to generate ideal playing strategy for SPLT.')

    parser.add_argument(
        '-r','--run_name',
        help="Will name the output directory for the results of the run",
        type=str,
        default="run",
    )

    return parser

def loadExperienceData(run_name):
    """
    Loads exists, or creates new experience data to use for experience replay.
    """
    folder_name = "dqn_experience_data"

    bet_mem_path = os.path.join(os.getcwd(),folder_name,run_name,"bet_experience_data.pkl")
    if os.path.exists(bet_mem_path):
        print("Loading existing bet experience data.")
        with open(bet_mem_path,'rb') as f:
            bet_experience_data = pickle.load(f)
    else:
        print("Previous bet experience data not found: generating empty memory list.")
        bet_experience_data = deque(maxlen=BET_EXPERIENCE_BANK_SIZE)

    eval_mem_path = os.path.join(os.getcwd(),folder_name,run_name,"eval_experience_data.pkl")
    if os.path.exists(eval_mem_path):
        print("Loading existing eval experience data.")
        with open(eval_mem_path,'rb') as f:
            eval_experience_data = pickle.load(f)
    else:
        print("Previous eval experience data not found: generating empty memory list.")
        eval_experience_data = deque(maxlen=EVAL_EXPERIENCE_BANK_SIZE)
    
    act_mem_path = os.path.join(os.getcwd(),folder_name,run_name,"act_experience_data.pkl")
    if os.path.exists(act_mem_path):
        print("Loading existing action experience data.")
        with open(act_mem_path,'rb') as f:
            act_experience_data = pickle.load(f)
    else:
        print("Previous action experience data not found: generating empty memory list.")
        act_experience_data = deque(maxlen=ACTION_EXPERIENCE_BANK_SIZE)

    return bet_experience_data, eval_experience_data, act_experience_data

def saveExperienceData(run_name,bet_exp_data,eval_exp_data,state_transition_bank):
    folder_name = "dqn_experience_data"

    bet_mem_path = os.path.join(os.getcwd(),folder_name,run_name,"bet_experience_data.pkl")
    with open(bet_mem_path,'wb') as f:
        print(f"Saving {len(bet_exp_data)} items of bet experience data in {bet_mem_path}")
        pickle.dump(bet_exp_data,f)
    
    eval_mem_path = os.path.join(os.getcwd(),folder_name,run_name,"eval_experience_data.pkl")
    with open(eval_mem_path,'wb') as f:
        print(f"Saving {len(eval_exp_data)} items of eval experience data in {eval_mem_path}")
        pickle.dump(eval_exp_data,f)
    
    act_mem_path = os.path.join(os.getcwd(),folder_name,run_name,"act_experience_data.pkl")
    with open(act_mem_path,'wb') as f:
        print(f"Saving {len(state_transition_bank)} items of state transition data in {act_mem_path}")
        pickle.dump(state_transition_bank,f)

def trainDQNAgent():
    parser = _build_parser()
    args = parser.parse_args()

    bet_exp_data, eval_exp_data, state_transition_bank = loadExperienceData(args.run_name)

    jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])
    #Wait until all experience banks are at least 1/4 full to start learning:
    while len(bet_exp_data)<BET_EXPERIENCE_BANK_SIZE/4 or len(eval_exp_data)<EVAL_EXPERIENCE_BANK_SIZE/4 \
        or len(state_transition_bank)<ACTION_EXPERIENCE_BANK_SIZE/4:
        bet_data, eval_data, state_transitions = jg.playGameAndTrackStateTransitions()

        #convert to form [np.array(n,a,b),np.array(n,c,d)] from multiple rows of [np.array(1,a,b),np.array(1,c,d)]
        # bet_data = postProcessBetTrainData(bet_data)
        # eval_data = postProcessTrainData(eval_data)

        #add to existing bet_exp_data bank
        bet_exp_data.extend(bet_data)
        eval_exp_data.extend(eval_data)
        state_transition_bank.extend(state_transitions)

        print(f"Bet: {len(bet_exp_data)}/{BET_EXPERIENCE_BANK_SIZE/4}, Eval: {len(eval_exp_data)}/{EVAL_EXPERIENCE_BANK_SIZE/4}, Act: {len(state_transition_bank)}/{ACTION_EXPERIENCE_BANK_SIZE/4}")

        jg.resetGame()
        #Need to be tracking next state, or reward if it is terminal.
        #playGameAndTrackData is actually collecting data we need for to train the value functions on
        #undiscounted monte carlo rewards, rather than actual one-step rewards.

        #tuple should now be (input state, input action, reward, next state)
        #Then when training on the reward later, recalculate expected value of next state, add to reward
        #How to implement gradient descent a la equation 3? Look at implementations in the literature.

    """
    Full algorithm:
    - generate at least a baseline amount of data
    - Loop forever:
        - play games epsilon random until 25000 state transitions have been generated, minibatches of 32 in between each step
        - Retrain bet and evaluation network on full amount of data: epochs 50, minibatch 256?
        - Run new agent for 5 games against all SimpleAgents, determine performance
        - If agent beats previous average score, keep it. Otherwise, revert back to old.
    """

    saveExperienceData(args.run_name, bet_exp_data, eval_exp_data, state_transition_bank)


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