#Judgement agent class
import random

from pkg_resources import invalid_marker
from deck_of_cards import deck_of_cards
from Situations import BetSituation, SubroundSituation

SUIT_ORDER = ["Spades","Hearts","Diamonds","Clubs","No Trump"]

class JudgementAgent(object):
    def __init__(self):
        self.points = 0
        self.hand = []
        self.available_cards = []
        self.subrounds_won = 0
        self.bet = -1
        self.id = 0

    def drawCard(self,card):
        self.hand.append(card)

    def determineCardOptions(self,srs):
        if len(srs.card_stack) > 0:
            forced_suit = srs.card_stack[0].suit
            cards_of_same_suit = [card for card in self.hand if card.suit == forced_suit]
        else:
            cards_of_same_suit = []

        if len(cards_of_same_suit) > 0:
            self.available_cards = cards_of_same_suit
        else:
            self.available_cards = self.hand

    def playCard(self,srs):
        """
        When called, determines indices of possible cards to play,
        and calls custom chooseCard method to choose card from list of possible cards.

        Based on SubroundSituation object.

        Returns choice of card (Card object)
        """
        self.determineCardOptions(srs)
        return self.chooseCard()
    
    def makeBet(self,bs):
        """
        Based on the given bet situation, chooses and sets a bet, and returns it.

        The base JudgementAgent class defines this function as a purely
        random choice, but in child objects this will usually take as input
        the gamestate and make a more intelligent choice of bet.

        Must make sure that any redefined makeBet method fulfills rulles of the game
        """
        possible_bets = list(range(bs.hand_size+1))
        if len(bs.other_bets) == (len(bs.agents)-1):
            invalid_bet = bs.hand_size-sum(bs.other_bets)
            if invalid_bet in possible_bets: possible_bets.remove(invalid_bet)
        self.bet = random.choice(possible_bets)
        return self.bet

    def chooseCard(self):
        """
        chooseCard method chooses a valid card to play and returns it.
        
        The base JudgementAgent class defines this function as a purely
        random choice, but in child objects this will usually take as input
        the gamestate and make a more intelligent choice.
        """
        card_to_play = random.choices(self.available_cards)[0]
        self.hand.remove(card_to_play)
        #returns chosen card and removes it from hand
        return card_to_play

    def updateScore(self,hand_size):
        #Betting zero
        if self.bet == 0:
            if self.subrounds_won == 0: self.points += 10*hand_size
            else: self.points += -10*self.subrounds_won
        
        else:
            #Didn't reach target (overbet)
            if self.bet > self.subrounds_won:
                self.points += -10*(self.bet-self.subrounds_won)
            #Got target exactly
            elif self.bet == self.subrounds_won:
                self.points += 10*self.bet
            #Went over target (underbet) (doesn't account for more players)
            else:
                rollover_num = 5
                old_ones_place = self.points % 10
                self.points += 10*self.bet + (self.subrounds_won-self.bet)
                new_ones_place = self.points % 10
                old_ones_place_5s = old_ones_place // rollover_num
                new_ones_place_5s = (old_ones_place+self.subrounds_won-self.bet) // rollover_num
                self.points += -rollover_num*10*(new_ones_place_5s-old_ones_place_5s)

    def reset(self):
        """ Resets agent to prepare for new round
        """
        self.hand = []
        self.available_cards = []
        self.subrounds_won = 0
        self.bet = -1

    def printHand(self):
        """
        prints hand in comprehensible way
        """
        print("{: <20}{: <20}{: <20}{: <20}".format(SUIT_ORDER[0],SUIT_ORDER[1],SUIT_ORDER[2],SUIT_ORDER[3]))
        suit_dict = {0:[],1:[],2:[],3:[]}
        for card in self.hand:
            suit_dict[card.suit].append(card)
        
        max_cards_in_suit = max([len(val) for val in suit_dict.values()])
        for row in range(max_cards_in_suit):
            for i in range(4):
                try:
                    print("{: <20}".format(suit_dict[i][row].name),end="")
                except IndexError:
                    print("{: <20}".format("--"),end="")
            print(" ")


if __name__ == "__main__":
    agent = JudgementAgent()

