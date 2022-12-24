"""
Initialize bet, evaluation, and model value functions based on supervised learning on games from HumanBetAgent.py
"""
from JudgmentGame import JudgmentGame
from JudgmentAgent import JudgmentAgent
from JudgmentValueModels import initBetModel, initActionModel, initEvalModel
from deck_of_cards import DeckOfCards
from nn_config import POINT_NORMALIZATION
from Situations import BetSituation, SubroundSituation
import numpy as np
from HumanBetAgent import HumanBetAgent
from collections import defaultdict as dd
from JudgmentUtils import calcSubroundAdjustedValue, convertBetSituationToBetState, convertSubroundSituationToActionState, convertSubroundSituationToEvalState
import os
import pickle
from math import floor
import time
from threading import Thread
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

SUIT_ORDER = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
DEFAULT_HAND_SIZES = [1,2,3,4,5,6,7,8,9,10,11,12,13,12,11,10,9,8,7,6,5,4,3,2,1]
DEFAULT_AGENTS = [JudgmentAgent(0),JudgmentAgent(1),JudgmentAgent(2),JudgmentAgent(3)]   
        
def postProcessTrainData(train_data_list):
    """
    Converts list of training data to batch form. Expects training data of the form:
    [[np.array(1,x,y),np.array(1,a,b)...],
     [np.array(1,x,y),np.array(1,a,b)...],
     ...,
     [np.array(1,x,y),np.array(1,a,b)...]]

    and converts to the form:
    [np.array(n,x,y),np.array(n,a,b)...]
    """
    example_train_data = train_data_list[0]
    
    #generate correctly sized numpy array
    reformatted_train_data = []
    batch_size = len(train_data_list)

    for example_train_data_element in example_train_data:
        new_shape = [batch_size] + list(np.shape(example_train_data_element))
        reformatted_train_data.append(np.zeros((new_shape)))

    print(example_train_data)
    print(example_train_data_element)

    #fill array with data
    for data_type_index in range(len(reformatted_train_data)):
        for i in range(batch_size):
            reformatted_train_data[data_type_index][i,:] = train_data_list[i][data_type_index]
        
    return reformatted_train_data

def postProcessBetTrainData(train_data_list):
    """
    Converts list of training data to batch form. Expects training data of the form:
    [[np.array(1,x,y),
     ...,
     [np.array(1,x,y)]

    and converts to the form:
    [np.array(n,x,y)]
    """
    example_train_data = train_data_list[0]
    
    #generate correctly sized numpy array
    batch_size = len(train_data_list)
    
    new_shape = [batch_size] + list(np.shape(example_train_data))
    reformatted_train_data = np.zeros((new_shape))

    #fill array with data
    for i in range(batch_size):
        reformatted_train_data[i,:] = train_data_list[i]
        
    return reformatted_train_data

def trainEvalFunctionOnExpertAlgorithm(use_old_data=True):
    #~~~~~~~~~~~ LOAD OR GENERATE EVAL TRAINING DATA
    eval_data_path = os.path.join(os.getcwd(),"eval_expert_train_data/eval_expert_train_data.pkl")
    if not use_old_data:
        start = time.time()
        eval_train_data = []

        train_data_goal = 100000
        last_modulus = np.inf
        while len(eval_train_data) < train_data_goal:
            jg = JudgmentGame(agents=[HumanBetAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)])

            bet_train_data, curr_eval_train_data, action_train_data = jg.playGameAndTrackData()
            eval_train_data += curr_eval_train_data
            print(f"Eval training data: {len(eval_train_data)}/{train_data_goal}")

            if len(eval_train_data) % floor(train_data_goal/5) < last_modulus:
                with open(eval_data_path, 'wb') as f:
                    print(f"Saving {len(eval_train_data)} pieces of training data in {eval_data_path}...")
                    pickle.dump(eval_train_data,f)
                last_modulus = len(eval_train_data) % floor(train_data_goal/5)
        
        with open(eval_data_path, 'wb') as f:
            print(f"Saving {len(eval_train_data)} pieces of training data in {eval_data_path}...")
            pickle.dump(eval_train_data,f)
        print(f"Done generating training data in {time.time()-start} seconds")
    else:
        with open(eval_data_path, 'rb') as f:
            eval_train_data = pickle.load(f)

    #~~~~~~~~~ SPLIT INTO TRAINING AND TEST SET~~~~~~~~~~~~~
    data_split = 0.8
    split_idx = int(data_split*len(eval_train_data))

    train_data_inputs = []
    train_data_outputs = []
    test_data_inputs = []
    test_data_outputs = []
    for i, train_data_total in enumerate(eval_train_data):
        if i < split_idx:
            train_data_inputs.append(train_data_total[0])
            train_data_outputs.append(int(train_data_total[1]))
        else:
            test_data_inputs.append(train_data_total[0])
            test_data_outputs.append(int(train_data_total[1]))
            
    train_data_inputs = postProcessTrainData(train_data_inputs)
    test_data_inputs = postProcessTrainData(test_data_inputs)
    train_data_outputs = np.array(train_data_outputs)
    test_data_outputs=np.array(test_data_outputs)

    model = initEvalModel()

    print("Begin Training:")
    start = time.time()
    model.fit(train_data_inputs,train_data_outputs,epochs=250,batch_size=128)
    print(f"Finished training in {time.time()-start} seconds:")

    print("Evaluation:")
    model.evaluate(test_data_inputs,test_data_outputs,verbose=2)

    eval_model_path = os.path.join(os.getcwd(),'eval_expert_train_model')
    model.save(eval_model_path)

def generateTrainingData(gen_eval_data=True,gen_act_data=True,gen_bet_data=True):
    """
    Generate training data from expert play to train agent action and bet network
    on expert games (eval network should be trained to predict expert games separately)
    """
    # def playGamesThread(index,training_games):
    #     game_num = 0
    #     bet_train_data = []
    #     action_train_data = []

    #     jg = JudgementGameWDataGen(agents=[HumanBetAgent(0,use_eval_model=True),HumanBetAgent(1,use_eval_model=True),HumanBetAgent(2,use_eval_model=True),HumanBetAgent(3,use_eval_model=True)])
    #     while game_num < training_games:
    #         curr_bet_train_data, eval_train_data, curr_action_train_data = jg.playGameAndTrackData()
    #         bet_train_data += curr_bet_train_data
    #         action_train_data += curr_action_train_data

    #         print(f"Thread {index}, Game {game_num}/{training_games}")
    #         game_num += 1
    #         jg.resetGame()

    #     bet_data_from_cores[index] = bet_train_data
    #     action_data_from_cores[index] = action_train_data

    # start = time.time()
    
    # bet_data_from_cores = [0,0,0,0]
    # action_data_from_cores = [0,0,0,0]

    # threads = []
    # for thread_num in range(2):
    #     process = Thread(target=playGamesThread, args=[thread_num,10])
    #     process.start()
    #     threads.append(process)
    # print(f"spawned threads in {time.time()-start} seconds")
    # for thread in threads:
    #     thread.join()
    #     print(f"Generated data in {time.time()-start} seconds")
    # print(len(bet_data_from_cores[0]))
    # print(len(bet_data_from_cores[1]))
    
    start = time.time()
    game_num = 0
    bet_train_data = []
    action_train_data = []
    eval_train_data = []
    training_games = 10
    jg = JudgmentGame(agents=[HumanBetAgent(0,use_eval_model=True),HumanBetAgent(1,use_eval_model=True),HumanBetAgent(2,use_eval_model=True),HumanBetAgent(3,use_eval_model=True)])
    while game_num < training_games:    
        curr_bet_train_data, curr_eval_train_data, curr_action_train_data = jg.playGameAndTrackData()
        if gen_bet_data: bet_train_data += curr_bet_train_data
        if gen_act_data: action_train_data += curr_action_train_data
        if gen_eval_data: eval_train_data += curr_eval_train_data

        print(f"Game {game_num}/{training_games}",end='\r')
        game_num += 1
        jg.resetGame()

    print(f"Generated data in {time.time()-start} seconds")

    bet_data_path = os.path.join(os.getcwd(),"bet_expert_train_data/bet_expert_train_data.pkl")
    action_data_path = os.path.join(os.getcwd(),"action_expert_train_data/action_expert_train_data.pkl")
    eval_data_path = os.path.join(os.getcwd(),"eval_expert_train_data/eval_expert_train_data.pkl")
    if gen_bet_data:
        with open(bet_data_path,'wb') as f:
            pickle.dump(bet_train_data,f)
    if gen_act_data:
        with open(action_data_path,'wb') as f:
            pickle.dump(action_train_data,f)
    if gen_eval_data:
        with open(eval_data_path,'wb') as f:
            pickle.dump(eval_train_data,f)

def trainModelOnExpertData(model_fn,data_path,model_path,epochs=250,batch_size=128):
    with open(data_path,'rb') as f:
        train_data = pickle.load(f)

    #~~~~~~~~~ SPLIT INTO TRAINING AND TEST SET~~~~~~~~~~~~~
    data_split = 0.8
    split_idx = int(data_split*len(train_data))

    train_data_inputs = []
    train_data_outputs = []
    test_data_inputs = []
    test_data_outputs = []
    for i, train_data_total in enumerate(train_data):
        if i < split_idx:
            train_data_inputs.append(train_data_total[0])
            train_data_outputs.append(int(train_data_total[1]))
        else:
            test_data_inputs.append(train_data_total[0])
            test_data_outputs.append(int(train_data_total[1]))
            
    train_data_inputs = postProcessTrainData(train_data_inputs)
    test_data_inputs = postProcessTrainData(test_data_inputs)
    train_data_outputs = np.array(train_data_outputs)
    test_data_outputs=np.array(test_data_outputs)

    model = model_fn()

    print("Begin Training:")
    start = time.time()
    model.fit(train_data_inputs,train_data_outputs,epochs=epochs,batch_size=batch_size)
    print(f"Finished training in {time.time()-start} seconds:")

    print("Evaluation:")
    model.evaluate(test_data_inputs,test_data_outputs,verbose=2)

    model.save(model_path)

def trainBetDataOnExpertData(epochs=250,batch_size=128):
    bet_data_path = os.path.join(os.getcwd(),"bet_expert_train_data/bet_expert_train_data.pkl")
    bet_model_path = os.path.join(os.getcwd(),"bet_expert_train_model")
    with open(bet_data_path,'rb') as f:
        train_data = pickle.load(f)

    #~~~~~~~~~ SPLIT INTO TRAINING AND TEST SET~~~~~~~~~~~~~
    data_split = 0.8
    split_idx = int(data_split*len(train_data))

    train_data_inputs = []
    train_data_outputs = []
    test_data_inputs = []
    test_data_outputs = []
    for i, train_data_total in enumerate(train_data):
        if i < split_idx:
            train_data_inputs.append(train_data_total[0])
            train_data_outputs.append(int(train_data_total[1]))
        else:
            test_data_inputs.append(train_data_total[0])
            test_data_outputs.append(int(train_data_total[1]))

    # print(train_data_inputs)      
    train_data_inputs = postProcessBetTrainData(train_data_inputs)
    test_data_inputs = postProcessBetTrainData(test_data_inputs)
    train_data_outputs = np.array(train_data_outputs)
    test_data_outputs=np.array(test_data_outputs)

    model = initBetModel()

    print("Begin Training:")
    start = time.time()
    model.fit(train_data_inputs,train_data_outputs,epochs=epochs,batch_size=batch_size)
    print(f"Finished training in {time.time()-start} seconds:")

    print("Evaluation:")
    model.evaluate(test_data_inputs,test_data_outputs,verbose=2)

    model.save(bet_model_path)

if __name__ == "__main__":
    # trainEvalFunctionOnExpertAlgorithm(use_old_data=False)\with open(eval_data_path, 'rb') as f:
    # generateTrainingData()
    trainBetDataOnExpertData()