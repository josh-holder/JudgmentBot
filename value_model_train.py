"""
Initialize bet, evaluation, and model value functions based on supervised learning on games from HumanBetAgent.py
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Masking
from tensorflow.keras import Input, Model
from JudgmentGame import JudgmentGame
from JudgmentAgent import JudgmentAgent
from deck_of_cards import DeckOfCards
from Situations import BetSituation, SubroundSituation
import numpy as np
from HumanBetAgent import HumanBetAgent
from collections import defaultdict as dd
from JudgmentUtils import calcSubroundAdjustedValue

SUIT_ORDER = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
DEFAULT_HAND_SIZES = [1,2,3,4,5,6,7,8,9,10,11,12,13,12,11,10,9,8,7,6,5,4,3,2,1]
DEFAULT_AGENTS = [JudgmentAgent(0),JudgmentAgent(1),JudgmentAgent(2),JudgmentAgent(3)]
POINT_NORMALIZATION = 130 #normalize by max realistic point change in a given round

def convertSubroundSituationToActionState(srs, agent, remaining_agents, publicly_played_cards, chosen_card, win_chance):
    """
    Given a SubroundSituation object, current Agent, a list of the Agents going after the current Agent, 
    a list of publicly available played cards (1 at index i if card i has been publicly played), and the chose card,
    outputs an action state suitable for input into the action Q function. Action state is of the form:

    [lstm_input for players still to go, 
    [current winner relative points, current winner bet, current winner earned (normalized),
    one-hot encoding of cards still in game, one-hot encoding of action cards, one-hot encoding of trump suit,
    agent bet, agent percentage of subrounds won, number of cards in hand]]
    """
    #~~~~~~Determine next_agents series to feed into LSTM~~~~~~~~~~
    next_agents_series = []
    for next_agent in remaining_agents:
        relative_points = (next_agent.points - agent.points)/POINT_NORMALIZATION
        next_agent_data = [relative_points,next_agent.bet/13,next_agent.subrounds_won/13] + next_agent.visibly_out_of_suit
        next_agents_series.append(next_agent_data)
    
    #pad next_agents_series with empty data which will be filtered out with a Mask layer later
    while len(next_agents_series) < 3:
        next_agents_series.append([0,0,0,0,0,0,0])

    winning_agent_state = [-1.0,-1.0,-1.0] #by default, there is no winning player so we have all -1s to be masked
    if len(srs.card_stack) > 0:
        #determine currently winning player
        for i,card in enumerate(srs.card_stack):
            if calcSubroundAdjustedValue(card,srs) == srs.highest_adjusted_val:
                winning_agent = srs.agents[i]
        
        #~~~~~~~~~winning agent information~~~~~~~~~~~~
        winning_agent_state[0] = (winning_agent.points - agent.points)/POINT_NORMALIZATION
        winning_agent_state[1] = winning_agent.bet/13
        winning_agent_state[2] = winning_agent.subrounds_won/13

    parameter_state = np.zeros(112)
    #~~~~~~~~~~cards still available information~~~~~~~~~~~~
    cards_still_available = np.ones(52,dtype=int)
    #if card is in publicly played cards, it can no longer be played by someone else
    cards_still_available = np.bitwise_xor(cards_still_available,publicly_played_cards)
    #if card is in your hand, it can't be played by someone else
    for card in agent.hand:
        cards_still_available[card.index] = 0
    cards_still_available[chosen_card.index] = 0 #the card you're about to play can't be played by someone else
    
    parameter_state[:52] = cards_still_available

    #~~~~~~~~~~~~card about to be played information~~~~~~~~~~~~
    card_to_be_played = np.zeros(52,dtype=int)
    card_to_be_played[chosen_card.index] = 1

    parameter_state[52:104] = card_to_be_played

    #~~~~~~~~~~~~one-hot trump encoding~~~~~~~~~~~~~~~~~~~~~~~~
    if srs.trump < 4: parameter_state[104+srs.trump] = 1 #if trump is 4, then there is no trump
    parameter_state[108] = agent.bet/13
    parameter_state[109] = agent.subrounds_won/13
    parameter_state[110] = win_chance
    parameter_state[111] = srs.hand_size

    return [next_agents_series, winning_agent_state, parameter_state]

def convertSubroundSituationToEvalState(srs, agent, remaining_agents, publicly_played_cards, chosen_card):
    """
    Input SubroundSituation object, current Agent, a list of the Agents going after the current Agent,
    a list of publicly played cards (1 at index i if card i has been publicly played), and the chosen Card.

    Returns None if the chosen card always loses or wins, as this is not a state where the neural network will be
    used. Otherwise, returns the evaluation state in the following form:

    [lstm_input for players still to go, 
    [one-hot encoding of cards still in game, one-hot encoding of trump suit, value of chosen card, 
    agent bet, agent percentage of subrounds won, number of cards in hand]]
    """
    adjusted_card_val = calcSubroundAdjustedValue(chosen_card,srs)

    if adjusted_card_val < srs.highest_adjusted_val: return None #always loses, don't need NN eval
    elif len(srs.card_stack) == 3 and adjusted_card_val > srs.highest_adjusted_val: return None #always wins, don't need NN eval
    else: #state which NN needs to predict
        #Determine next_agents series to feed into LSTM
        next_agents_series = []
        for next_agent in remaining_agents:
            relative_points = (next_agent.points - agent.points)/POINT_NORMALIZATION
            next_agent_data = [relative_points,next_agent.bet/13,next_agent.subrounds_won/13] + next_agent.visibly_out_of_suit
            next_agents_series.append(next_agent_data)

        #pad next_agents_series with empty data which will be filtered out with a Mask layer later
        while len(next_agents_series) < 3:
            next_agents_series.append([0,0,0,0,0,0,0])

        #Determine parameter state
        parameter_state = np.zeros(60)
        cards_still_available = np.ones(52,dtype=int)
        #if card is in publicly played cards, it can no longer be played by someone else
        cards_still_available = np.bitwise_xor(cards_still_available,publicly_played_cards)
        #if card is in your hand, it can't be played by someone else
        for card in agent.hand:
            cards_still_available[card.index] = 0
        cards_still_available[chosen_card.index] = 0 #the card you're about to play can't be played by someone else

        parameter_state[:52] = cards_still_available
        if srs.trump < 4: parameter_state[52+srs.trump] = 1 #if trump is 4, then there is no trump
        parameter_state[56] = adjusted_card_val/26
        parameter_state[57] = agent.bet/13
        parameter_state[58] = agent.subrounds_won/13
        parameter_state[59] = srs.hand_size

        return [next_agents_series, parameter_state]


def convertBetSituationToBetState(bs, hand, bet):
    """
    Given bet situation, hand, and bet value, outputs a bet state suitable for input into
    the bet Q function. Bet state is of the form:

    [one-hot encoding of hand - bet position - existence of zero bet - % of bets already taken - one-hot encoding of trump - bet amount]
    """
    bet_state = np.zeros(60)
    #add cards to bet data point
    for card in hand:
        data_index = 13*card.suit + card.value - 1
        bet_state[card.index] = 1
    bet_state[52] = len(bs.other_bets)
    bet_state[53] = 1 if 0 in bs.other_bets else 0 #1 if there's been a zero bet
    bet_state[54] = sum(bs.other_bets)/bs.hand_size #sum of previous bets divided by max. Can divide by hand b/c no cards have been played
    if bs.trump < 4: bet_state[55+bs.trump] = 1 #if trump is 4, then there is no trump

    bet_state[59] = bet

    return bet_state


class JudgementGameWDataGen(JudgmentGame):
    def playGameForData(self):
        """
        Carries out simulated judgement game.

        For each hand size in self.hand_sizes,
        Deals cards to each agent, then initiates betting,
        and then incites Agents to play subrounds until no cards remain.
        Upon completion, distributes points to each agent accordingly.

        Tallies points cumulatively and returns a list of agents.
        """
        bet_train_data = [] #list of tuples with (bet_input_state, bet_value)
        action_train_data = []
        eval_train_data = []
        for round, hand_size in enumerate(self.hand_sizes):
            bet_state_input_data = {} #keys: agent ids. values: bet states, which will later be paired with an average score difference
            eval_state_input_data = {}
            action_state_input_data = dd(list)

            self.resetAgents()
            trump = round % len(SUIT_ORDER)

            if self.game_verbose: print("~~ Round {}, Hand Size {}, Trump {} ~~".format(round,hand_size,SUIT_ORDER[trump]))

            deck = DeckOfCards()

            #~~~~~~~~~~~~~~DEAL CARDS~~~~~~~~~~~~~~~
            for card_num in range(hand_size):
                for agent in self.agents:
                    agent.drawCard(deck.give_random_card())
            
            #~~~~~~~~~~~~~INITIATE BETS, COLLECT BET STATE DATA~~~~~~~~~~~~~~~~
            bets = []
            bs = BetSituation(hand_size,bets,trump,self.agents)
            for agent in self.agents:                
                #make bet
                agent_bet = agent.makeBet(bs)
                bets.append(agent_bet)
                bet_state_input_data[agent.id] = convertBetSituationToBetState(bs,agent.hand,agent_bet)

                bs.other_bets = bets
                
            if self.game_verbose: print("Grabbed bets: {}".format(bets))

            #~~~~~~~~~~~~~~PLAY CARDS FROM HAND, COLLECT EVAL AND ACTION DATA~~~~~~~~~~~~~~~
            publicly_played_cards = np.zeros(52,dtype=int)
            starting_agent = 0
            turn_order = self.agents
            for subround in range(hand_size):
                #set new turn order based on who won last round
                turn_order = turn_order[starting_agent:]+turn_order[:starting_agent]
                srs = SubroundSituation(hand_size,[],trump,0,turn_order)

                #Each agent plays a card from it's hand
                for agent_ind, agent in enumerate(turn_order):
                    chosen_card = agent.playCard(srs)
                    
                    win_chance = agent.evalSubroundWinChance(chosen_card,srs)
                    remaining_agents = turn_order[agent_ind+1:]

                    action_state_input_data[agent.id].append(convertSubroundSituationToActionState(srs, agent, remaining_agents, publicly_played_cards, chosen_card, win_chance))
                    eval_state_input_data[agent.id] = convertSubroundSituationToEvalState(srs,agent,remaining_agents,publicly_played_cards,chosen_card)

                    srs.highest_adjusted_val = max(srs.highest_adjusted_val, calcSubroundAdjustedValue(chosen_card, srs))
                    srs.card_stack.append(chosen_card)
                    publicly_played_cards[chosen_card.index] = 1

                winning_agent_ind = self.evaluateSubround(srs)
                winning_agent_id = turn_order[winning_agent_ind].id
                turn_order[winning_agent_ind].subrounds_won += 1
                starting_agent = winning_agent_ind

                for agent in self.agents:
                    if eval_state_input_data[agent.id] != None:
                        eval_train_data.append((eval_state_input_data[agent.id],agent.id==winning_agent_id))

                if self.game_verbose: 
                    print("Subround {}, order is now {} {} {} {} - cards played were:".format(subround,turn_order[0].id,turn_order[1].id,turn_order[2].id,turn_order[3].id))
                    for card in srs.card_stack:
                        print(card.name,end=", ")
                    print("\nWinning card is: {}".format(srs.card_stack[winning_agent_ind].name))

                if self.print_final_tables:
                    for agent in self.agents:
                        try:
                            agent.displayTable(srs)
                            print(f"Player {winning_agent_id} won trick with {srs.card_stack[winning_agent_ind].name}")
                            input("Press any key to continue")
                        #If agent doesn't have a way to print the table (i.e. is not the Human player)
                        except AttributeError:
                            pass

            if self.game_verbose:
                for agent in self.agents:
                    print("Agent {} bet {}, won {}, had {} points.".format(agent.id,agent.bet,agent.subrounds_won,agent.points))

            #update points
            point_changes = []
            for agent in self.agents:
                point_change = agent.updateScore(hand_size)
                point_changes.append(point_change)
            
            #Add associated point changes to each bet and action input state:
            for point_change, agent in zip(point_changes,self.agents):
                avg_point_change_of_others = (sum(point_changes)-point_change)/3
                point_change_difference = (point_change-avg_point_change_of_others)/POINT_NORMALIZATION

                bet_state = bet_state_input_data[agent.id]
                bet_train_data.append((bet_state,point_change_difference))

                for action_state in action_state_input_data[agent.id]:
                    action_train_data.append((action_state,point_change_difference))

            if self.game_verbose:
                for agent in self.agents:
                    print("Thus, new agent score is {}.".format(agent.points))
                print(" ")

            self.agents.append(self.agents.pop(0)) #shift order of agents for next round

        print(action_train_data)
        self.agents = sorted(self.agents, key=lambda x: x.id)
        print(len(action_train_data))
        print(len(eval_train_data))
        print(len(bet_train_data))
        return bet_train_data, eval_train_data, action_train_data
        
def initBetModel():
    """
    Model which, given bet situation, will output the expected average value of the bet relative to the
    other players (normalized by 260, which is the maximum point difference).
    """
    model = tf.keras.models.Sequential([
        keras.Input(shape=(60)),
        layers.Dense(32,activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16,activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(16,activation='relu'),
        layers.Dense(1,activation='sigmoid'),
    ])
    model.compile(loss = tf.losses.MeanSquaredError(),
                    optimizer = tf.optimizers.Adam())
    return model

def initEvalModel(layer_sizes=[48,24,12]):
    """
    Model which, given playing situation, outputs the probability that playing a given card will result
    in winning the hand.

    Inputs:
    -LSTM output of the players going after you in the order. Each timestep in the LSTM contains the following info:
        - Player relative points
        - Player bet
        - Player earned (normalized)
        - Suits remaining (4 binary values)
    - Cards still possible to play (one-hot encoding with 52 binary values)
    - Trump suit (4 binary values, also one-hot encoding)
    - Value of card to play, representing action (0 if off-suit, 1-13 if secondary suit, 14-26 if trump suit)
    - Agent bet (normalized to 13)
    - Agent subrounds already won (normalized to 13)
    - Number of cards in hand
    """
    #Input for LSTM of players going after you in the order
    next_players_input = Input(shape=(3,7))
    next_players_mask = Masking()(next_players_input)
    next_players_LSTM = layers.LSTM(20, activation="relu", unroll=True)(next_players_mask)

    #Input for all other parameters, including cards still possible to play, trump, player info, etc.
    parameters = Input(shape=60)

    #Concatenate LSTM and regular parameters into fully connected layers
    all_params = concatenate([next_players_LSTM, parameters])

    #Add the specified amount and size of fully connected layers
    input_layer = all_params
    for i,layer in enumerate(layer_sizes):
        MLP_layer = layers.Dense(48,activation="relu")(input_layer)
        if i % 2 == 0:
            adjust_layer = layers.Dropout(0.1)(MLP_layer)
        else:
            adjust_layer = layers.BatchNormalization()(MLP_layer)

        input_layer = adjust_layer

    #Predict percentage win value
    win_value = layers.Dense(1,activation="sigmoid")(input_layer)

    model = Model(inputs=[next_players_input,parameters],outputs=win_value)
    model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

    return model

def initActionModel(layer_sizes=[64, 64, 48,24,12]):
    """
    Model which, given a playing situation, outputs the expected values of all the cards.

    Inputs:
    -LSTM output of the players going after you in the order. Each timestep in the LSTM contains the following info:
        - Player relative points
        - Player bet
        - Player earned (normalized)
        - Suits remaining (4 binary values)
    -Information on the current winner of the round
        - Player relative points
        - Player bet
        - Player earned (normalized)
    - Cards still possible to play (one-hot encoding with 52 binary values)
    - Card that you're about to play (one-hot encoding with 52 binary values)
    - Trump suit (4 binary values, also one-hot encoding)
    - Agent bet (normalized to 13)
    - Agent subrounds already won (normalized to 13)
    - Chance that that card is going to win
    - Number of cards in hand

    Output:
    Expected value of the action relative to other players, normalized to 260.
    """
    #Input for LSTM of players going after you in the order
    next_players_input = Input(shape=(3,7))
    next_players_mask = Masking()(next_players_input)
    next_players_LSTM = layers.LSTM(20, activation="relu", unroll=True)(next_players_mask)

    #Input for currently winning player (need masking here if there is no winning player)
    winning_player_input = Input(shape=(3))
    winning_player_mask = Masking(mask_value=-1.0)(winning_player_input) #masking value can't be 0 because all zeros is plausible here
    
    #Input for other parameters (cards not played, action, trump, agent info, etc.)
    parameters = Input(shape=109)

    #Concatenate LSTM, winning player, and regular parameters into fully connected layers
    all_params = concatenate([next_players_LSTM, winning_player_mask, parameters])

    #Add the specified amount and size of fully connected layers
    input_layer = all_params
    for i,layer in enumerate(layer_sizes):
        MLP_layer = layers.Dense(48,activation="relu")(input_layer)
        if i % 2 == 0:
            adjust_layer = layers.Dropout(0.1)(MLP_layer)
        else:
            adjust_layer = layers.BatchNormalization()(MLP_layer)

        input_layer = adjust_layer

    #Predict action value
    action_value = layers.Dense(1,activation="sigmoid")(input_layer)

    model = Model(inputs=[next_players_input,winning_player_input,parameters],outputs=action_value)
    model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

    return model

def generateTrainingData():
    jg = JudgmentGame()

if __name__ == "__main__":
    jg = JudgementGameWDataGen(game_verbose=True,agents=[HumanBetAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)])

    jg.playGameForData()

