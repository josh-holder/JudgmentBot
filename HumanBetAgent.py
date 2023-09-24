from SimpleAgent import SimpleAgent
from judgment_data_utils import convertSubroundSituationToEvalState
import os
from tensorflow import keras
import tensorflow as tf
from bet_nn_train import prepareData
import numpy as np
import time

class HumanBetAgent(SimpleAgent):
    def __init__(self,id,bet_model_name="human_bet_model",eval_model_name="eval_expert_train_model",use_eval_model=False):
        super().__init__(id)
        bet_model_path = os.path.join(os.getcwd(),bet_model_name)
        self.bet_model = keras.models.load_model(bet_model_path)

        #load evaluation model trained on data from HumanBet agents - updated 12/22
        self.use_eval_model = use_eval_model
        if use_eval_model:
            eval_model_path = os.path.join(os.getcwd(),eval_model_name)
            self.eval_model = keras.models.load_model(eval_model_path)

    def convertBetSituationToNNInput(self,bs):
        bs_as_nn_input = -0.5*np.ones((1,56)) #want mean to be approx 0

        #add cards
        for card in self.hand:
            data_index = 13*card.suit + card.value - 2
            bs_as_nn_input[0,data_index] = 0.5

        bs_as_nn_input[0,52] = len(bs.other_bets)
        bs_as_nn_input[0,53] = bs.other_bets.count(0)
        bs_as_nn_input[0,54] = sum(bs.other_bets)/bs.hand_size
        bs_as_nn_input[0,55] = bs.trump

        return bs_as_nn_input

    def makeBet(self, bs):
        """
        Bets using trained NN
        """
        # start = time.time()
        # model_path = os.path.join(os.getcwd(),"bet_agent")
        # bet_model = keras.models.load_model(model_path)
        # print("Bet time {}".format(time.time()-start))
        
        bs_as_nn_input = self.convertBetSituationToNNInput(bs)

        percent_of_cards = self.bet_model(bs_as_nn_input)

        desired_bet = percent_of_cards*bs.hand_size

        possible_bets = list(range(bs.hand_size+1))
        if len(bs.other_bets) == (len(bs.agents)-1):
            invalid_bet = bs.hand_size-sum(bs.other_bets)
            if invalid_bet in possible_bets: possible_bets.remove(invalid_bet)

        #find closes bet in possible bets to desired
        
        min_dist_from_poss = np.inf
        for possible_bet in possible_bets:
            dist = abs(desired_bet-possible_bet)
            if dist < min_dist_from_poss:
                min_dist_from_poss = dist
                self.bet = possible_bet

        return self.bet

    def evalSubroundWinChance(self, srs, card):
        basic_output = super().evalSubroundWinChance(srs, card)

        #If the deterministic version of the function couldn't come up with an answer,
        #determine the win chance with a neural network.
        if basic_output == None:
            if self.use_eval_model:
                eval_state = convertSubroundSituationToEvalState(srs, self, card)
                eval_state = [eval_state_component[np.newaxis,:] for eval_state_component in eval_state]
                return self.eval_model(eval_state)
            else: return None
        else: 
            return basic_output


if __name__ == "__main__":
    pass