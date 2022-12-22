class BetSituation(object):
    def __init__(self,hand_size,other_bets,trump,agents):
        self.hand_size = hand_size
        self.other_bets = other_bets
        self.trump = trump
        self.agents = agents

class SubroundSituation(object):
    """
    Aims to be a self-contained object which contains all the information needed to
    fully define a subround sitation. Using this info combined with info about it's own hand,
    any agent can determine what card to play in a given situation.
    """
    def __init__(self,hand_size,card_stack,trump,highest_adjusted_val,agents,publicly_played_cards):
        self.hand_size = hand_size #Used as input to NN. Useful to determine how bad a 0 bet would be for you, for example.
        self.card_stack = card_stack #Used to determine forced suit, for example
        self.trump = trump
        self.highest_adjusted_val = highest_adjusted_val #Saves computation by saving the highest adjusted value card on the stack
        self.agents = agents #Characteristics of both winning and future agents are useful in determining correct action
        self.publicly_played_cards = publicly_played_cards #Useful for determining which cards can still be played in the subround