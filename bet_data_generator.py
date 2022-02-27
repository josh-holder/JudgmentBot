from deck_of_cards import deck_of_cards
import os
import pickle
import argparse
import random
import time

def _build_parser():
	parser = argparse.ArgumentParser(description='Generate betting data to train neural network Judgment better.')

	parser.add_argument(
		'-b','--bet_data_name',
		help="Set name of output file/file to load for betting data. If existing file, will append new bet data",
		type=str,
		default="bet_data/bet_data.pkl",
	)

	return parser

def loadBetData(folder_path="bet_data"):
    combined_data = []
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename != ".DS_Store":
            with open(f,'rb') as f:
                dat = pickle.load(f)
                combined_data.extend(dat)
    
    return combined_data


class betSituation(object):
    def __init__(self,round):
        self.round = round
        self.deck = deck_of_cards.DeckOfCards()
        self.cards = []
        for i in range(self.round):
            self.cards.append(self.deck.give_random_card())
        self.sortCardsJudgment()
        self.bet_pos = random.randint(0,3)
        self.other_bets = self.genOtherBets()
        self.trump = (self.round-1) % 5

    def genOtherBets(self):
        """
        Need a way of generating plausible other bets - if you just pick bets completely
        randomly, they will never appear like normal judgement bets and training data will
        be significantly decreased.

        Model with bias towards betting zero (because high reward), bias against betting
        high when high bets have already been made, and vice versa.

        Intuitively, the model performs pretty realistically with these weights
        """
        bets_so_far = 0
        base_probs = [10 for i in range(self.round+1)]
        base_probs[0] = 12.5 #bias for zero

        other_bets = []
        for bet_ind in range(self.bet_pos):
            bets_left = self.round-bets_so_far
            probs = []
            for bet_num in range(self.round+1):
                if bets_left > 0:
                    too_high_bet_adjust = bet_num/bets_left
                else: #need this to avoid divide by zero and handle situations where we've already overbet
                    too_high_bet_adjust = (bets_so_far+bet_num)/self.round
                probs.append(max(base_probs[bet_num]-5*(too_high_bet_adjust)*(self.round/4)-3*((bets_left-bet_num)/self.round)*(bet_ind/3),0))
            bet = random.choices(range(self.round+1),weights=probs)[0]
            bets_so_far += bet
            other_bets.append(bet)
        
        return other_bets
    
    def sortCardsJudgment(self):
        for card in self.cards:
            if card.value == 1: card.value = 14#change aces to high
        self.cards.sort(key=lambda x: x.value, reverse=True)
        self.cards.sort(key=lambda x: x.suit)

    def printHand(self):
        """
        prints hand in comprehensible way
        """
        suits = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
        suits[self.trump] = "~"+suits[self.trump]+"~"
        print("{: <20}{: <20}{: <20}{: <20}".format(suits[0],suits[1],suits[2],suits[3]))
        suit_dict = {0:[],1:[],2:[],3:[]}
        for card in self.cards:
            suit_dict[card.suit].append(card)
        
        max_cards_in_suit = max([len(val) for val in suit_dict.values()])
        for row in range(max_cards_in_suit):
            for i in range(4):
                try:
                    print("{: <20}".format(suit_dict[i][row].name),end="")
                except IndexError:
                    print("{: <20}".format("--"),end="")
            print(" ")

    def printSituationInfo(self):
        suits = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
        suit_name = suits[self.trump]
        print("---Round {}, {}---".format(bs.round,suit_name))
        for i,bet in enumerate(self.other_bets):
            print("Bet {}: {}".format(i+1,bet))
        
        if len(self.other_bets) == 0:
            print("Betting first!")
        
        cant_bet = self.round - sum(self.other_bets)
        if len(self.other_bets) == 3 and sum(self.other_bets) <= self.round:
            print("Can't bet {}".format(cant_bet))

        print("")
        self.printHand()


def requestUserBet(bs):

    bs.printSituationInfo()    

    input_success = False
    while not input_success:
        bet = input("Bet:")
        try: 
            if not(int(bet) == (bs.round - sum(bs.other_bets)) and bs.bet_pos == 3) and int(bet) <= bs.round:
                bet = int(bet)
                input_success = True
        except ValueError:
            print("Try again - invalid bet")
    
    return bet
    

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    bet_data_path = os.path.join(os.getcwd(),args.bet_data_name)
    if os.path.exists(bet_data_path):
        print("Loading existing bet data to append to:")
        with open(bet_data_path,'rb') as f:
            bet_data = pickle.load(f)
    else:
        print("Creating bet data file with name {}".format(args.bet_data_name))
        bet_data = []

    bet_num = 0
    while True:
        bs = betSituation(random.randint(1,13))
        bet = requestUserBet(bs)
        bet_data.append((bs,bet))

        print(" ")
        print("~Bet recorded - {} bets have been generated~".format(len(bet_data)))
        print(" ")
        time.sleep(1.5)

        if bet_num % 5 == 0:
            print("Saving data...")
            with open(bet_data_path,'wb') as f:
                pickle.dump(bet_data,f)
            print("Saved!")
        
        bet_num += 1