from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from JudgmentUtils import calcSubroundAdjustedValue
import os
from tensorflow import keras
import tensorflow as tf
from bet_nn_train import prepareData
import numpy as np
import time

class HumanBetAgent(SimpleAgent):
    def __init__(self,id,bet_model_name="human_bet_model"):
        super().__init__(id)
        bet_model_path = os.path.join(os.getcwd(),bet_model_name)
        self.bet_model = keras.models.load_model(bet_model_path)

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


if __name__ == "__main__":
    pass