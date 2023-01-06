from JudgmentAgent import JudgmentAgent
from deck_of_cards import DeckOfCards
from Situations import BetSituation, SubroundSituation
from SimpleAgent import SimpleAgent
from HumanAgent import HumanAgent
from HumanBetAgent import HumanBetAgent
from DQNAgent import copyDQNAgentsWithoutModels, DQNAgent
from JudgmentUtils import calcSubroundAdjustedValue, convertBetSituationToBetState, convertSubroundSituationToActionState, convertSubroundSituationToEvalState
import time
import numpy as np
from nn_config import POINT_NORMALIZATION
from collections import defaultdict as dd
from copy import deepcopy

SUIT_ORDER = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
DEFAULT_HAND_SIZES = [1,2,3,4,5,6,7,8,9,10,11,12,13,12,11,10,9,8,7,6,5,4,3,2,1]
DEFAULT_AGENTS = [JudgmentAgent(0),JudgmentAgent(1),JudgmentAgent(2),JudgmentAgent(3)]

class JudgmentGame(object):
    def __init__(self,agents=DEFAULT_AGENTS,hand_sizes=DEFAULT_HAND_SIZES,game_verbose=0,print_final_tables=0):
        self.hand_sizes = hand_sizes
        self.print_final_tables = print_final_tables
        self.agents = agents
        self.game_verbose = game_verbose

    def playGame(self):
        """
        Carries out simulated judgement game.

        For each hand size in self.hand_sizes,
        Deals cards to each agent, then initiates betting,
        and then incites Agents to play subrounds until no cards remain.
        Upon completion, distributes points to each agent accordingly.

        Tallies points cumulatively and returns a list of agents.
        """
        for round, hand_size in enumerate(self.hand_sizes):
            self.resetAgents()
            trump = round % len(SUIT_ORDER)

            if self.game_verbose: print("~~ Round {}, Hand Size {}, Trump {} ~~".format(round,hand_size,SUIT_ORDER[trump]))

            deck = DeckOfCards()

            #deal cards
            for card_num in range(hand_size):
                for agent in self.agents:
                    agent.drawCard(deck.give_random_card())
            
            
            #initiate bets
            bets = []
            bs = BetSituation(hand_size,bets,trump,self.agents)
            for agent in self.agents:
                bets.append(agent.makeBet(bs))
                bs.other_bets = bets

            if self.game_verbose: print("Grabbed bets: {}".format(bets))

            starting_agent = 0
            turn_order = self.agents
            srs = SubroundSituation(hand_size,[],trump,0,turn_order,np.zeros(52,dtype=int))
            for subround in range(hand_size):
                #set new turn order based on who won last round
                turn_order = turn_order[starting_agent:]+turn_order[:starting_agent]
                srs.card_stack = []
                srs.highest_adjusted_val = 0
                srs.agents = turn_order

                #Each agent plays a card from it's hand
                for agent in turn_order:
                    chosen_card = agent.playCard(srs)

                    srs.highest_adjusted_val = max(srs.highest_adjusted_val, calcSubroundAdjustedValue(chosen_card, srs))
                    srs.card_stack.append(chosen_card)
                    srs.publicly_played_cards[chosen_card.index] = 1

                winning_agent_ind = self.evaluateSubround(srs)
                turn_order[winning_agent_ind].subrounds_won += 1
                starting_agent = winning_agent_ind

                if self.game_verbose: 
                    print("Subround {}, order is now {} {} {} {} - cards played were:".format(subround,turn_order[0].id,turn_order[1].id,turn_order[2].id,turn_order[3].id))
                    for card in srs.card_stack:
                        print(card.name,end=", ")
                    print("\nWinning card is: {}".format(srs.card_stack[winning_agent_ind].name))

                if self.print_final_tables:
                    for agent in self.agents:
                        try:
                            agent.displayTable(srs)
                            print(f"Player {self.agents[winning_agent_ind].id} won trick with {srs.card_stack[winning_agent_ind].name}")
                            input("Press any key to continue")
                        #If agent doesn't have a way to print the table (i.e. is not the Human player)
                        except AttributeError:
                            pass

            if self.game_verbose:
                for agent in self.agents:
                    print("Agent {} bet {}, won {}, had {} points.".format(agent.id,agent.bet,agent.subrounds_won,agent.points))

            #update points
            for agent in self.agents:
                agent.updateScore(hand_size)

            if self.game_verbose:
                for agent in self.agents:
                    print("Thus, new agent score is {}.".format(agent.points))
                print(" ")

            self.agents.append(self.agents.pop(0)) #shift order of agents for next round
        
        self.agents = sorted(self.agents, key=lambda x: x.id)

        return self.agents

    def playGameAndCollectSLData(self):
        """
        Carries out simulated judgement game.

        Carries out judgement game identically to playGame, except it tracks and stores
        training data from the runs to use in supervised learning.

        i.e. stores (action_state, point differential experienced at end of subround.)
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
                bet_state_input_data[agent.id] = convertBetSituationToBetState(bs,agent,agent_bet)
                bs.other_bets = bets
                
            if self.game_verbose: print("Grabbed bets: {}".format(bets))

            #~~~~~~~~~~~~~~PLAY CARDS FROM HAND, COLLECT EVAL AND ACTION DATA~~~~~~~~~~~~~~~
            starting_agent = 0
            turn_order = self.agents
            srs = SubroundSituation(hand_size,[],trump,0,copyDQNAgentsWithoutModels(turn_order),np.zeros(52,dtype=int))
            for subround in range(hand_size):
                #set new turn order based on who won last round
                turn_order = turn_order[starting_agent:]+turn_order[:starting_agent]
                srs.card_stack = []
                srs.highest_adjusted_val = 0
                srs.agents = copyDQNAgentsWithoutModels(turn_order)

                #Each agent plays a card from it's hand
                for agent in turn_order:
                    chosen_card = agent.playCard(srs)

                    action_state_input_data[agent.id].append(convertSubroundSituationToActionState(srs, agent, chosen_card))
                    eval_state_input_data[agent.id] = convertSubroundSituationToEvalState(srs, agent, chosen_card)

                    srs.highest_adjusted_val = max(srs.highest_adjusted_val, calcSubroundAdjustedValue(chosen_card, srs))
                    srs.card_stack.append(chosen_card)
                    srs.publicly_played_cards[chosen_card.index] = 1

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

        return bet_train_data, eval_train_data, action_train_data

    def playGameAndTrackStateTransitions(self):
        """
        Plays games of Judgement identically to playGame(), but stores state transitions and rewards for actions,
        and supervised learning data for betting and evaluation.

        Returns lists of tuples of state transitions of the form:
        action_transitions = [(init_srs, init_action, srs1),
                                (srs1, action1, srs2),
                                ...
                                (final_srs, final_action, reward)]
        *Note that this ends with a reward, not a SubroundSituation in the final slot, indicating that this is
        the terminal state. This is possible because only the terminal state has any reward. If you're trying to make
        these rewards non-sparse, you'll need to add a new column storing rewards associated with each transition.
        
        Bets and evaluation data will be stored as before, as supervised learning data of the form:
        bet_sl_data = [(bs, reward),
                        (bs, reward),
                        ...
                        (bs, reward)]
        """
        state_transitions = []
        bet_sl_data = [] #list of tuples with (bet_input_state, bet_value)
        eval_sl_data = []
        for round, hand_size in enumerate(self.hand_sizes):
            bet_state_input_data = {} #keys: agent ids. values: bet states, which will later be paired with an average score difference
            eval_state_input_data = {}
            #key: agent ids. values: history of SubroundSituations for subround. 
            subround_situation_transition_data = dd(list)
            action_transition_data = dd(list)

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
                bet_state_input_data[agent.id] = convertBetSituationToBetState(bs,agent,agent_bet)

                bs.other_bets = bets
                
            if self.game_verbose: print("Grabbed bets: {}".format(bets))

            #~~~~~~~~~~~~~~PLAY CARDS FROM HAND, COLLECT EVAL AND ACTION DATA~~~~~~~~~~~~~~~
            starting_agent = 0
            turn_order = self.agents
            srs = SubroundSituation(hand_size,[],trump,0,copyDQNAgentsWithoutModels(turn_order),np.zeros(52,dtype=int))
            for subround in range(hand_size):
                #set new turn order based on who won last round
                turn_order = turn_order[starting_agent:]+turn_order[:starting_agent]
                srs.card_stack = []
                srs.highest_adjusted_val = 0
                srs.agents = copyDQNAgentsWithoutModels(turn_order)

                #Each agent plays a card from it's hand
                for agent in turn_order:
                    subround_situation_transition_data[agent.id].append(deepcopy(srs))

                    chosen_card = agent.playCard(srs)
                    action_transition_data[agent.id].append(chosen_card)

                    eval_state_input_data[agent.id] = convertSubroundSituationToEvalState(srs, agent, chosen_card)

                    srs.highest_adjusted_val = max(srs.highest_adjusted_val, calcSubroundAdjustedValue(chosen_card, srs))
                    srs.card_stack.append(chosen_card)
                    srs.publicly_played_cards[chosen_card.index] = 1

                winning_agent_ind = self.evaluateSubround(srs)
                winning_agent_id = turn_order[winning_agent_ind].id
                turn_order[winning_agent_ind].subrounds_won += 1
                starting_agent = winning_agent_ind

                for agent in self.agents:
                    if eval_state_input_data[agent.id] != None:
                        eval_sl_data.append((eval_state_input_data[agent.id],agent.id==winning_agent_id))

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
                #normalize this by the max realistic value of points, because activation function is tanh and thus [-1,1]
                point_change_difference = (point_change-avg_point_change_of_others)/POINT_NORMALIZATION

                bet_state = bet_state_input_data[agent.id]
                bet_sl_data.append((bet_state,point_change_difference))

                situations_transferring_from = subround_situation_transition_data[agent.id]
                #The states that you're transitioning to are shifted forward by 1, with the final point change as the reward at the end
                situations_transferring_to = subround_situation_transition_data[agent.id][1:] + [point_change_difference]
                for srs_from, action_from, srs_to in zip(situations_transferring_from, action_transition_data[agent.id], situations_transferring_to):
                    state_transitions.append((srs_from, action_from, srs_to))

            if self.game_verbose:
                for agent in self.agents:
                    print("Thus, new agent score is {}.".format(agent.points))
                print(" ")

            self.agents.append(self.agents.pop(0)) #shift order of agents for next round
        # print(state_transitions)
        return bet_sl_data, eval_sl_data, state_transitions

    def evaluateSubround(self,srs):
        """
        Given a stack of cards, evaluates which card won,
        and returns that index.
        """
        #Shift the values of any trump card up by 13 and
        #keep the values of secondary trump cards the same.
        #Set values of other suits to zero, and simply pick the card
        #with the highest value
        card_values = []
        for card in srs.card_stack:
            card_values.append(calcSubroundAdjustedValue(card,srs))

        #return index in card_values where the card of maximum value was found
        return card_values.index(max(card_values))


    def resetAgents(self):
        """
        Resets hands, bets, subrounds won, available cards of all
        agents in game. Does not reset points.
        """
        for agent in self.agents:
            agent.reset()

    def resetGame(self):
        """
        Resets all properties of agents, including points.
        """
        for agent in self.agents:
            agent.reset()
            agent.points=0


if __name__ == "__main__":
    # scores = [0,0,0,0]
    # for game_num in range(25):
    #     jg = JudgmentGame(game_verbose=1,agents=[SimpleAgent(0),JudgmentAgent(1),JudgmentAgent(2),JudgmentAgent(3)])
    #     agents = jg.playGame()
    #     for i,agent in enumerate(agents):
    #         scores[i] += agent.points

    # print("Final Scores: {}".format([score for score in scores]))

    jg = JudgmentGame(game_verbose=1,agents=[DQNAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)])
    jg.playGame()