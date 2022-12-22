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
import threading
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

SUIT_ORDER = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
DEFAULT_HAND_SIZES = [1,2,3,4,5,6,7,8,9,10,11,12,13,12,11,10,9,8,7,6,5,4,3,2,1]
DEFAULT_AGENTS = [JudgmentAgent(0),JudgmentAgent(1),JudgmentAgent(2),JudgmentAgent(3)]

class JudgementGameWDataGen(JudgmentGame):
    def playGameForData(self):
        """
        Carries out simulated judgement game.

        For each hand size in self.hand_sizes,
        Deals cards to each agent, then initiates betting,
        and then incites Agents to play subrounds until no cards remain.
        Upon completion, distributes points to each agent accordingly.

        Tallies points cumulatively and returns a list of agents.
        """
        bet_train_data = [] #list of tuples with (bet_input_state, bet_value)
        action_train_data = []
        eval_train_data = []
        for round, hand_size in enumerate(self.hand_sizes):
            bet_state_input_data = {} #keys: agent ids. values: bet states, which will later be paired with an average score difference
            eval_state_input_data = {}
            action_state_input_data = dd(list)

            self.resetAgents()
            trump = round % len(SUIT_ORDER)

            if self.game_verbose: print("~~ Round {}, Hand Size {}, Trump {} ~~".format(round,hand_size,SUIT_ORDER[trump]))

            deck = DeckOfCards()

            #~~~~~~~~~~~~~~DEAL CARDS~~~~~~~~~~~~~~~
            for card_num in range(hand_size):
                for agent in self.agents:
                    agent.drawCard(deck.give_random_card())
            
            #~~~~~~~~~~~~~INITIATE BETS, COLLECT BET STATE DATA~~~~~~~~~~~~~~~~
            bets = []
            bs = BetSituation(hand_size,bets,trump,self.agents)
            for agent in self.agents:                
                #make bet
                agent_bet = agent.makeBet(bs)
                bets.append(agent_bet)
                bet_state_input_data[agent.id] = convertBetSituationToBetState(bs,agent,agent_bet)

                bs.other_bets = bets
                
            if self.game_verbose: print("Grabbed bets: {}".format(bets))

            #~~~~~~~~~~~~~~PLAY CARDS FROM HAND, COLLECT EVAL AND ACTION DATA~~~~~~~~~~~~~~~
            starting_agent = 0
            turn_order = self.agents
            srs = SubroundSituation(hand_size,[],trump,0,turn_order,np.zeros(52,dtype=int))
            for subround in range(hand_size):
                #set new turn order based on who won last round
                turn_order = turn_order[starting_agent:]+turn_order[:starting_agent]
                srs.card_stack = []
                srs.highest_adjusted_val = 0
                srs.agents = turn_order

                #Each agent plays a card from it's hand
                for agent_ind, agent in enumerate(turn_order):
                    chosen_card = agent.playCard(srs)

                    action_state_input_data[agent.id].append(convertSubroundSituationToActionState(srs, agent, chosen_card))
                    eval_state_input_data[agent.id] = convertSubroundSituationToEvalState(srs, agent, chosen_card)

                    srs.highest_adjusted_val = max(srs.highest_adjusted_val, calcSubroundAdjustedValue(chosen_card, srs))
                    srs.card_stack.append(chosen_card)
                    srs.publicly_played_cards[chosen_card.index] = 1

                winning_agent_ind = self.evaluateSubround(srs)
                winning_agent_id = turn_order[winning_agent_ind].id
                turn_order[winning_agent_ind].subrounds_won += 1
                starting_agent = winning_agent_ind

                for agent in self.agents:
                    if eval_state_input_data[agent.id] != None:
                        eval_train_data.append((eval_state_input_data[agent.id],agent.id==winning_agent_id))

                if self.game_verbose: 
                    print("Subround {}, order is now {} {} {} {} - cards played were:".format(subround,turn_order[0].id,turn_order[1].id,turn_order[2].id,turn_order[3].id))
                    for card in srs.card_stack:
                        print(card.name,end=", ")
                    print("\nWinning card is: {}".format(srs.card_stack[winning_agent_ind].name))

                if self.print_final_tables:
                    for agent in self.agents:
                        try:
                            agent.displayTable(srs)
                            print(f"Player {winning_agent_id} won trick with {srs.card_stack[winning_agent_ind].name}")
                            input("Press any key to continue")
                        #If agent doesn't have a way to print the table (i.e. is not the Human player)
                        except AttributeError:
                            pass

            if self.game_verbose:
                for agent in self.agents:
                    print("Agent {} bet {}, won {}, had {} points.".format(agent.id,agent.bet,agent.subrounds_won,agent.points))

            #update points
            point_changes = []
            for agent in self.agents:
                point_change = agent.updateScore(hand_size)
                point_changes.append(point_change)
            
            #Add associated point changes to each bet and action input state:
            for point_change, agent in zip(point_changes,self.agents):
                avg_point_change_of_others = (sum(point_changes)-point_change)/3
                point_change_difference = (point_change-avg_point_change_of_others)/POINT_NORMALIZATION

                bet_state = bet_state_input_data[agent.id]
                bet_train_data.append((bet_state,point_change_difference))

                for action_state in action_state_input_data[agent.id]:
                    action_train_data.append((action_state,point_change_difference))

            if self.game_verbose:
                for agent in self.agents:
                    print("Thus, new agent score is {}.".format(agent.points))
                print(" ")

            self.agents.append(self.agents.pop(0)) #shift order of agents for next round

        return bet_train_data, eval_train_data, action_train_data
        

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
    
    #fill array with data
    for data_type_index in range(len(reformatted_train_data)):
        for i in range(batch_size):
            reformatted_train_data[data_type_index][i,:] = train_data_list[i][data_type_index]
        
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
            jg = JudgementGameWDataGen(agents=[HumanBetAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)])

            bet_train_data, curr_eval_train_data, action_train_data = jg.playGameForData()
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

def generateTrainingData():
    """
    Generate training data from expert play to train agent action and bet network
    on expert games (eval network should be trained to predict expert games separately)
    """
    training_games = 1000
    game_num = 0
    bet_train_data = []
    action_train_data = []
    start = time.time()
    
    jg = JudgementGameWDataGen(agents=[HumanBetAgent(0,use_eval_model=True),HumanBetAgent(1,use_eval_model=True),HumanBetAgent(2,use_eval_model=True),HumanBetAgent(3,use_eval_model=True)])
    while game_num < training_games:    
        curr_bet_train_data, eval_train_data, curr_action_train_data = jg.playGameForData()
        bet_train_data += curr_bet_train_data
        action_train_data += curr_action_train_data

        print(f"Game {game_num}/{training_games}",end='\r')
        game_num += 1
        jg.resetGame()

    print(f"Generated data in {time.time()-start} seconds")

    bet_data_path = os.path.join(os.getcwd(),"bet_expert_train_data/bet_expert_train_data.pkl")
    action_data_path = os.path.join(os.getcwd(),"action_expert_train_data/action_expert_train_data.pkl")
    with open(bet_data_path,'wb') as f:
        pickle.dump(bet_train_data,f)
    with open(action_data_path,'wb') as f:
        pickle.dump(action_train_data,f)
    
    print(len(action_train_data))
    print(len(bet_train_data))

if __name__ == "__main__":
    # trainEvalFunctionOnExpertAlgorithm(use_old_data=False)
    generateTrainingData()
