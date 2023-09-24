#Judgment agent class
import random
from judgment_data_utils import calcSubroundAdjustedValue

SUIT_ORDER = ["Spades","Hearts","Diamonds","Clubs","No Trump"]

class JudgmentAgent(object):
    def __init__(self,id):
        self.points = 0
        self.hand = [] #List of Card objects
        self.available_cards = []
        self.subrounds_won = 0
        self.bet = -1
        self.id = id
        #structure which tracks whether other players know that this agent is out of a suit
        self.visibly_out_of_suit = [0,0,0,0] 

    def drawCard(self,card):
        self.hand.append(card)

    def determineCardOptions(self,srs):
        """
        Based on cards in play, edit available_cards object of Agent.
        """
        if len(srs.card_stack) > 0:
            forced_suit = srs.card_stack[0].suit
            cards_of_same_suit = [card for card in self.hand if card.suit == forced_suit]
            if len(cards_of_same_suit) == 0: self.visibly_out_of_suit[forced_suit] = 1
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
        chosen_card = self.chooseCard(srs)

        self.hand.remove(chosen_card)
        return chosen_card
    
    def makeBet(self,bs):
        """
        Based on the given bet situation, chooses and sets a bet, and returns it.

        The base JudgmentAgent class defines this function as a purely
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

    def chooseCard(self,srs):
        """
        chooseCard method chooses a valid card to play and returns it.
        
        The base JudgmentAgent class defines this function as a purely
        random choice, but in child objects this will usually take as input
        the gamestate and make a more intelligent choice.
        """
        card_to_play = random.choices(self.available_cards)[0]
        return card_to_play

    def evalSubroundWinChance(self,srs,card):
        """
        Given card (usually the card that the agent chose to play) and a SubroundSituation, 
        determines the chance that a card has to win the game.

        In basic implementation, returns a 1 or a 0 if it is deterministic, and None if it
        depends on the behavior of other agents. In other agents, win chance in these cases may be
        determined by a more complex algorithm, such as a neural network.
        """
        adjusted_card_val = calcSubroundAdjustedValue(card,srs)
        #If the card already loses to an existing card on the stack, win_chance is zero
        if adjusted_card_val < srs.highest_adjusted_val: return 0
        #Otherwise, if all other players have already played a card and this card is higher, win chance is 1
        elif len(srs.card_stack) == 3 and adjusted_card_val > srs.highest_adjusted_val: return 1
        #Otherwise, we can't determine the win chance with this basic method:
        else: return None

    def updateScore(self,hand_size):
        #Betting zero
        if self.bet == 0:
            if self.subrounds_won == 0:
                point_change = 10*hand_size
                self.points += point_change
                return point_change
            else: 
                point_change = -10*self.subrounds_won
                self.points += point_change
                return point_change
        
        else:
            #Didn't reach target (overbet)
            if self.bet > self.subrounds_won:
                point_change = -10*(self.bet-self.subrounds_won)
                self.points += point_change
                return point_change
            #Got target exactly
            elif self.bet == self.subrounds_won:
                point_change = 10*self.bet
                self.points += point_change
                return point_change
            #Went over target (underbet) (doesn't account for more players)
            else:
                rollover_num = 5
                old_ones_place = self.points % 10
                point_change = 10*self.bet + (self.subrounds_won-self.bet)
                new_ones_place = self.points % 10
                old_ones_place_5s = old_ones_place // rollover_num
                new_ones_place_5s = (old_ones_place+self.subrounds_won-self.bet) // rollover_num
                point_change += -rollover_num*10*(new_ones_place_5s-old_ones_place_5s)
                self.points += point_change
                return point_change

    def reset(self):
        """
        Resets agent to prepare for new round
        """
        self.hand = []
        self.available_cards = []
        self.subrounds_won = 0
        self.bet = -1
        self.visibly_out_of_suit = [0,0,0,0]

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
    agent = JudgmentAgent()

