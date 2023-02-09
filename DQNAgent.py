from JudgmentUtils import convertSubroundSituationToEvalState, convertSubroundSituationToActionState, convertBetSituationToBetState
from SimpleAgent import SimpleAgent
from HumanBetAgent import HumanBetAgent
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Masking
from tensorflow.keras import Input, Model
import numpy as np
import os
import random
import copy
import pickle

def copyDQNAgentsWithoutModels(init_agents):
    """
    Given a list of DQN agents, copies them and returns them without their models attached.
    Does this to save computation when copying models.
    """
    new_agents = []
    for init_agent in init_agents:
        new_agent = DQNAgent(init_agent.id,load_models=False)
        new_agent.points = init_agent.points
        new_agent.hand = copy.deepcopy(init_agent.hand)
        new_agent.subrounds_won = init_agent.subrounds_won
        new_agent.bet = init_agent.bet
        new_agent.visibly_out_of_suit = copy.copy(init_agent.visibly_out_of_suit)
        new_agent.id = init_agent.id

        new_agents.append(new_agent)

    return new_agents

class DQNAgent(SimpleAgent):
    def __init__(self,id,epsilon=0,load_models=True,bet_model_name="bet_expert_train_model",eval_model_name="eval_expert_train_model",action_model_name="action_expert_train_model"):
        super().__init__(id)
        self.epsilon = epsilon

        if load_models:
            bet_model_path = os.path.join(os.getcwd(),bet_model_name)
            self.bet_model = keras.models.load_model(bet_model_path)

            eval_model_path = os.path.join(os.getcwd(),eval_model_name)
            self.eval_model = keras.models.load_model(eval_model_path)

            action_model_path = os.path.join(os.getcwd(),action_model_name)
            self.action_model = keras.models.load_model(action_model_path)
        else:
            self.action_model = None
            self.bet_model = None
            self.eval_model = None

    def evalSubroundWinChance(self,srs,card):
        """
        Given a card and a subround, evaluates the chance that playing a given card will win the round.
        Deterministically defines win chance as zero if the given action is playing a card lower
        that the current highest card value.
        Otherwise, uses a NN representation to determine this.

        Inputs:
        - Action
        - Cards still possible to play (one-hot encoding with 52 binary values)
        - Trump suit (4 binary values, also one-hot encoding)
        - Highest card played (0 if off-suit, 1-13 if secondary suit, 14-26 if trump suit)
        - Agent bet
        - Agent subrounds already won
        - Sequence of information on player remaining, including:
            - Player relative points
            - Player bet
            - Player earned (normalized)
            - Suits remaining (4 binary values)

        Outputs:
        - List of probabilities of winning the subround, with index i correpsonding to the probability
            of winning the subround with card i in the agent's hand.
        """   
        basic_output = super().evalSubroundWinChance(srs, card)

        #If the deterministic version of the function couldn't come up with an answer,
        #determine the win chance with a neural network.
        if basic_output == None:
            eval_state = convertSubroundSituationToEvalState(srs, self, card)
            eval_state = [eval_state_component[np.newaxis,:] for eval_state_component in eval_state]
            return self.eval_model(eval_state)
        else: 
            return basic_output

    def chooseCard(self, srs):
        """
        Given a subround situation, determines what card to play:

        --EVALUATION Q() FUNCTION:--
        State input information:
        - cards still possible to play (52)
        - trump (4)
        - Player played info - RNN architecture
            - value of card played (26) (0 if off-suit, 1-13 if secondary suit, 14-26 if trump suit)
            - player relative points (1)
            - player bet (1)
            - player earned (1)
            - suits remaining (4)
        - Players remaining architecture - RNN architecture
            - player relative points (1)
            - player bet (1)
            - player earned (1)
            - suits remaining (4)
        -Player bet
        -Player earned

        Actions - 52 cards in hand (see connect 4 implementation for removal of impossible moves)
        
        Neural network learns action value function for particular state-action pair - 0 if loss, 1 if win given a particular card.
        
        After the end of every round, evaluate loss on what you predicted vs what you ended up with
        Good reward signal!
        --------------------------
        ACTION Q FUNCTION - predicted reward of given action (in terms of actual points)
        Inputs:
        - Probabilities of winning with every card in hand from evaluation network
        - player bet
        - player earned
        - cards still possible to play (52)
        - trump (4)

        (or just two heads, trained slightly differently? probably this)
        (Read AlphaGo paper)

        Output: card to play from hand

        --BET MODEL (Action value) ---------
        - same inputs as current bet model
        -(+points, look at transfer learning) (current data is zero point differential, retrain)
        ------------------------------------

        Model is given.
        First start off-policy with SimpleAgent epsilon-greedy actions,
        epsilon-greedy Neural-Network based bet actions to kickstart training.
        -After each timestep, run gradient descent on evaluation Q function
        -After each round, run gradient descent on action Q function
        -After each round, run gradient descent on bet Q function
        Run until near even with HumanBetAgent.
        Then, run fully model-free for finetuning until completion.

        Questions:
        - How to go about training RNNs for input into evaluation function
        - How to go about adding multiple heads to one function, using one in the other?
        """
        best_card = None
        best_act_val = -np.inf
        for card in self.available_cards:
            act_state = convertSubroundSituationToActionState(srs,self,card)

            act_state = [act_state_component[np.newaxis,:] for act_state_component in act_state]
            act_val = self.action_model(act_state)

            if act_val > best_act_val:
                best_card = card
                best_act_val = act_val

        #If epsilon is not zero, select action epsilon-greedily.
        if self.epsilon > 0:
            rand_num = random.random()
            num_valid_actions = len(self.available_cards)

            threshold = 0
            old_threshold = 0

            for card in self.available_cards:
                if card == best_card:
                    threshold += (1-self.epsilon)+self.epsilon/num_valid_actions
                else:
                    threshold += self.epsilon/num_valid_actions

                if old_threshold <= rand_num and rand_num <= threshold:
                    return card
                else:
                    old_threshold=threshold

            raise Exception(f"ERROR: Agent {self.id} failed to select a card.")

        #If epsilon=0, return greedy action
        else:
            return best_card


    def makeBet(self, bs):
        possible_bets = list(range(bs.hand_size+1))
        if len(bs.other_bets) == (len(bs.agents)-1):
            invalid_bet = bs.hand_size-sum(bs.other_bets)
            if invalid_bet in possible_bets: possible_bets.remove(invalid_bet)

        best_bet = None
        best_bet_val = -np.inf
        for bet in possible_bets:
            bet_state = convertBetSituationToBetState(bs, self, bet)
            # bet_state = [bet_state_component[np.newaxis,:] for bet_state_component in bet_state]
            bet_state = bet_state[np.newaxis,:]
            bet_val = self.bet_model(bet_state)

            if bet_val > best_bet_val:
                best_bet = bet
                best_bet_val = bet_val
            #If the bet is larger than 4 and it's not better than the last one, stop evaluating
            elif bet > 4:
                break
        
        #If epsilon is not zero, select bet epsilon-greedily.
        if self.epsilon > 0:
            rand_num = random.random()
            num_valid_bets = len(possible_bets)

            threshold = 0
            old_threshold = 0

            for bet in possible_bets:
                if bet == best_bet:
                    threshold += (1-self.epsilon)+self.epsilon/num_valid_bets
                else:
                    threshold += self.epsilon/num_valid_bets

                if old_threshold <= rand_num and rand_num <= threshold:
                    self.bet = bet
                    return self.bet
                else:
                    old_threshold=threshold
            
            raise Exception(f"ERROR: Agent {self.id} failed to make a bet.")
        
        #If epsilon=0, return greedy bet
        else:
            self.bet = best_bet
            return self.bet