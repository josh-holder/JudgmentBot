from JudgmentAgent import JudgmentAgent
import random
from JudgmentUtils import calcSubroundAdjustedValue

play_verbose = 0

class SimpleAgent(JudgmentAgent):
    """
    Agent with very simple betting and playing strategy.

    Scores approximately -25 points on average playing with all random opponents,
    who score on average -450 points
    """
    def makeBet(self, bs):
        """
        Very simple betting scheme makes it so that if many rounds are already accounted for,
        doesn't bet as many cards.

        Think most of the value is that it makes betting extremely high less common, and thus
        makes the bot lose less rounds.
        """
        bets_so_far = sum(bs.other_bets)
        bet_ind = len(bs.other_bets)
        base_probs = [10 for i in range(bs.hand_size+1)]
        base_probs[0] = 12.5 #bias for zero

        bets_left = bs.hand_size-bets_so_far
        probs = []
        for bet_num in range(bs.hand_size+1):
            if bets_left > 0:
                too_high_bet_adjust = bet_num/bets_left
            else: #need this to avoid divide by zero and handle situations where we've already overbet
                too_high_bet_adjust = (bets_so_far+bet_num)/bs.hand_size
            probs.append(max(base_probs[bet_num]-5*(too_high_bet_adjust)*(bs.hand_size/4)-3*((bets_left-bet_num)/bs.hand_size)*(bet_ind/3),0.01))
        self.bet = random.choices(range(bs.hand_size+1),weights=probs)[0]
        
        return self.bet

    def chooseCard(self,srs):
        #if other people have played before you
        if len(srs.card_stack) != 0:
            secondary_trump = srs.card_stack[0].suit
            if play_verbose: 
                print("Betting with {} cards before - secondary trump is {}".format(len(srs.card_stack),secondary_trump))
                print("Bet: {}, already won {}, stack is: ".format(self.bet,self.subrounds_won))
                for card in srs.card_stack:
                    print(card.name,end=" ")
                print("\nHand is:")
                self.printHand()

            #convert card stack to adjusted values of cards
            card_stack_values = []
            for card in srs.card_stack:
                card_stack_values.append(calcSubroundAdjustedValue(card,srs))

            #convert your cards to values
            hand_values = []
            for card in self.available_cards:
                if calcSubroundAdjustedValue(card,srs) == 0:
                    #ensure that the algorithm knows that some cards are more valubale than others, even if they're both worth zero this round
                    hand_values.append(card.value*0.1) 
                else:
                    hand_values.append(calcSubroundAdjustedValue(card,srs))

            currently_winning_value = max(card_stack_values)
            high_card_index = hand_values.index(max(hand_values))
            low_card_index = hand_values.index(min(hand_values))
            high_card = self.available_cards[high_card_index]
            low_card = self.available_cards[low_card_index]

            if play_verbose: print("High card {}, low card {}".format(high_card.name,low_card.name))

            #if you haven't reached your goal yet
            if self.bet > self.subrounds_won:
                
                #if your highest card is currently higher, play it
                if high_card.value > currently_winning_value: 
                    card_to_play = high_card
                    if play_verbose: print("Haven't reached goal, high card can beat {}, playing high card".format(currently_winning_value))
                #if you can't beat the current high card, play your lowest card
                if high_card.value <= currently_winning_value: 
                    card_to_play = low_card
                    if play_verbose: print("Haven't reached goal, high card can't beat {}, playing low card".format(currently_winning_value))

            #if you have reached your goal, play the highest card you can
            elif self.bet <= self.subrounds_won:
                cards_that_lose = []
                for card, adjusted_value in zip(self.available_cards,hand_values):
                    if adjusted_value < currently_winning_value:
                        cards_that_lose.append(card)

                if len(cards_that_lose) != 0:
                    max_value_of_card_that_loses = -1
                    for card in cards_that_lose:
                        if card.value > max_value_of_card_that_loses:
                            losing_card = card
                            max_value_of_card_that_loses = card.value
                    
                    card_to_play = losing_card

                #if all cards win, play the highest
                else:
                    card_to_play = high_card

        #if you're the first one playing
        elif len(srs.card_stack) == 0:
            if play_verbose: 
                print("Playing first:")
                print("Bet: {}, already won {}".format(self.bet,self.subrounds_won))
                print("\nHand is:")
                self.printHand()


            #if you're still trying to get bets, play your highest card (not considering suit)
            hand_values = []
            for card in self.hand:
                hand_values.append(card.value)

            high_card_index = hand_values.index(max(hand_values))
            low_card_index = hand_values.index(min(hand_values))
            high_card = self.available_cards[high_card_index]
            low_card = self.available_cards[low_card_index]
            if self.bet > self.subrounds_won:
                if play_verbose: "Need to win more, so playing high card"
                card_to_play = high_card
            elif self.bet <= self.subrounds_won: #otherwise, play the low card
                if play_verbose: "At or above bet already, so playing low card"
                card_to_play = low_card

        if play_verbose: print("Playing {}".format(card_to_play.name))

        return card_to_play