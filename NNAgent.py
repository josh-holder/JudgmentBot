from judgment_data_utils import convertSubroundSituationToEvalState, convertSubroundSituationToActionState, convertBetSituationToBetState
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

def copyNNAgentsWithoutModels(init_agents):
    """
    Given a list of NN agents, copies them and returns them without their models attached.
    Does this to save computation when copying models.
    """
    new_agents = []
    for init_agent in init_agents:
        new_agent = NNAgent(init_agent.id,load_models=False)
        new_agent.points = init_agent.points
        new_agent.hand = copy.deepcopy(init_agent.hand)
        new_agent.subrounds_won = init_agent.subrounds_won
        new_agent.bet = init_agent.bet
        new_agent.visibly_out_of_suit = copy.copy(init_agent.visibly_out_of_suit)
        new_agent.id = init_agent.id
        new_agent.epsilon = init_agent.epsilon

        new_agents.append(new_agent)

    return new_agents

class NNAgent(SimpleAgent):
    def __init__(self,id,epsilon=0,load_models=True,bet_model_name="bet_expert_train_model",eval_model_name="eval_expert_train_model",action_model_name="action_expert_train_model"):
        super().__init__(id)
        self.epsilon = epsilon

        if load_models:
            self.loadModels(bet_model_name, eval_model_name, action_model_name)
        else:
            self.action_model = None
            self.bet_model = None
            self.eval_model = None

    def loadModels(self, bet_model_name, eval_model_name, action_model_name):
        bet_model_path = os.path.join(os.getcwd(),bet_model_name)
        self.bet_model = keras.models.load_model(bet_model_path, compile=False)
        self.bet_model.compile()

        eval_model_path = os.path.join(os.getcwd(),eval_model_name)
        self.eval_model = keras.models.load_model(eval_model_path, compile=False)
        self.eval_model.compile()

        action_model_path = os.path.join(os.getcwd(),action_model_name)
        self.action_model = keras.models.load_model(action_model_path, compile=False)
        self.action_model.compile()

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
        Given a subround situation, determines what card to play,
        either epsilon-greedily or greedily, and returns it.
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
        
    def chooseCardAndReturnNetworkEvals(self, srs):
        """
        Given a subround situation, determines what card to play,
        either epsilon-greedily or greedily, and returns it.

        Also returns the NN evaluation of the subround situation for
        the chosen card, so we can use that in computing gradients
        in A3C without calling the NN again.
        """
        best_card = None
        best_act_val = -np.inf
        best_card_eval = None
        card_evals = []
        card_act_vals = []
        for card in self.available_cards:
            act_state = convertSubroundSituationToActionState(srs,self,card)

            act_state = [act_state_component[np.newaxis,:] for act_state_component in act_state]
            act_val = self.action_model(act_state)

            card_evals.append(float(act_state[2][:,110]))
            card_act_vals.append(act_val)

            if act_val > best_act_val:
                best_card = card
                best_act_val = act_val
                best_card_eval = float(act_state[2][:,110])

        #If epsilon is not zero, select action epsilon-greedily.
        if self.epsilon > 0:
            rand_num = random.random()
            num_valid_actions = len(self.available_cards)

            threshold = 0
            old_threshold = 0

            for card, card_act_val, card_eval in zip(self.available_cards, card_act_vals, card_evals):
                if card == best_card:
                    threshold += (1-self.epsilon)+self.epsilon/num_valid_actions
                else:
                    threshold += self.epsilon/num_valid_actions

                if old_threshold <= rand_num and rand_num <= threshold:
                    return card, card_act_val, card_eval
                else:
                    old_threshold=threshold

            raise Exception(f"ERROR: Agent {self.id} failed to select a card.")

        #If epsilon=0, return greedy action
        else:
            return best_card, best_act_val, best_card_eval

    def playCardAndReturnNetworkEvals(self,srs):
        self.determineCardOptions(srs)
        chosen_card, card_act_val, card_eval = self.chooseCardAndReturnNetworkEvals(srs)

        self.hand.remove(chosen_card)
        return chosen_card, card_act_val, card_eval
    
    def makeBetAndReturnNetworkEval(self, bs):
        possible_bets = list(range(bs.hand_size+1))
        if len(bs.other_bets) == (len(bs.agents)-1):
            invalid_bet = bs.hand_size-sum(bs.other_bets)
            if invalid_bet in possible_bets: possible_bets.remove(invalid_bet)

        best_bet = None
        best_bet_val = -np.inf

        bet_evals = []
        for bet in possible_bets:
            bet_state = convertBetSituationToBetState(bs, self, bet)
            # bet_state = [bet_state_component[np.newaxis,:] for bet_state_component in bet_state]
            bet_state = bet_state[np.newaxis,:]
            bet_val = self.bet_model(bet_state)

            bet_evals.append(bet_val)

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

            for bet_num, bet in enumerate(possible_bets):
                if bet == best_bet:
                    threshold += (1-self.epsilon)+self.epsilon/num_valid_bets
                else:
                    threshold += self.epsilon/num_valid_bets

                if old_threshold <= rand_num and rand_num <= threshold:
                    self.bet = bet

                    #If we hadn't previously evaluated the value of the bet (because it was over 4)
                    #then we need to evaluate and return it now
                    if bet_num >= len(bet_evals):
                        bet_state = convertBetSituationToBetState(bs, self, bet)
                        # bet_state = [bet_state_component[np.newaxis,:] for bet_state_component in bet_state]
                        bet_state = bet_state[np.newaxis,:]
                        bet_val = self.bet_model(bet_state)

                        return self.bet, bet_val
                    #Otherwise, return the bet and it's evaluation
                    else:
                        return self.bet, bet_evals[bet_num]
                else:
                    old_threshold=threshold
            
            raise Exception(f"ERROR: Agent {self.id} failed to make a bet.")
        
        #If epsilon=0, return greedy bet
        else:
            self.bet = best_bet
            return self.bet, best_bet_val