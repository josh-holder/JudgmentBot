class BetSituation(object):
    def __init__(self,hand_size,other_bets,trump,agents):
        self.hand_size = hand_size
        self.other_bets = other_bets
        self.trump = trump
        self.agents = agents

class SubroundSituation(object):
    def __init__(self,card_stack,trump,agents):
        self.card_stack = card_stack
        self.trump = trump
        self.agents = agents