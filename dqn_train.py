from collections import deque
import pickle
import os
from nn_config import ACTION_EXPERIENCE_BANK_SIZE, BET_EXPERIENCE_BANK_SIZE, EVAL_EXPERIENCE_BANK_SIZE
import argparse

from JudgmentGame import JudgmentGame
from DQNAgent import DQNAgent

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

def addToExperienceBank(exp_data, game_data):
    """
    Add data from current game to it's respective experience bank.

    TODO
    """

    pass


def trainDQNAgent():
    parser = _build_parser()
    args = parser.parse_args()

    bet_exp_data, eval_exp_data, act_exp_data = loadExperienceData(args.run_name)

    jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])
    #Wait until all experience banks are at least 1/4 full to start learning:
    while len(bet_exp_data)<BET_EXPERIENCE_BANK_SIZE/4 or len(eval_exp_data)<EVAL_EXPERIENCE_BANK_SIZE/4 or len(act_exp_data)<ACTION_EXPERIENCE_BANK_SIZE/4:
        bet_game_data, eval_game_data, action_game_data = jg.playGameAndTrackData()
        #Need to be tracking next state, or reward if it is terminal.
        #playGameAndTrackData is actually collecting data we need for to train the value functions on
        #undiscounted monte carlo rewards, rather than actual one-step rewards.

        #tuple should now be (input state, input action, reward, next state)
        #Then when training on the reward later, recalculate expected value of next state, add to reward
        #How to implement gradient descent a la equation 3? Look at implementations in the literature.


if __name__ == "__main__":
    trainDQNAgent()    