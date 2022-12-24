from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from HumanBetAgent import HumanBetAgent
import random
play_verbose = 0

class HumanAgent(JudgmentAgent):
    """
    Agent which allows human to play against bots
    """
    def displayTable(self,srs):
        """
        Based on the subround situation (cards played, points, bets, etc.),
        displays the state of the table ahead of the Human player playing a card.
        """
        # display top players hand (which will be two away in order from human player)
        #only works in 4 player games
        player_index = "Not found"
        for i,agent in enumerate(srs.agents):
            if agent.id == self.id:
                player_index = i
                break
        
        top_player_index = (player_index + 2) % 4
        right_player_index = (player_index + 3) % 4
        left_player_index = (player_index + 1) % 4
        top_player = srs.agents[top_player_index]
        left_player = srs.agents[left_player_index]
        right_player = srs.agents[right_player_index]
        print("{: <30}Player {}, {} points".format(" ",top_player.id,top_player.points))
        print("{: <30}Earned {}/{}".format(" ",top_player.subrounds_won,top_player.bet))
        print(" ")
        print("{: <30}_______________________".format(" "))
        print("{: <30}|Cards in play:       |".format(" "))
        for line_num in range(4):
            to_print = ""
            if line_num < len(srs.card_stack):
                center_str = "|{: <21}|      ".format(srs.card_stack[line_num].name.split(": ")[-1])
            else:
                center_str = "|{: <21}|      ".format("--")
            if line_num == 0:
                left_player_text = "{: <30}".format(f"Player {left_player.id}, {left_player.points} points")
                right_player_text = "{: <30}".format(f"Player {right_player.id}, {right_player.points} points")
                to_print = left_player_text + center_str +right_player_text
            elif line_num == 1:
                left_player_text = "{: <30}".format(f"Earned {left_player.subrounds_won}/{left_player.bet}")
                right_player_text = "{: <30}".format(f"Earned {right_player.subrounds_won}/{right_player.bet}")
                to_print = left_player_text + center_str + right_player_text
            else:
                to_print = "{: <30}".format(" ") + center_str + "{: <30}".format(" ")
            print(to_print)

        print("{: <30}_______________________       ".format(" "))
        print("")
        print("{: <30}You, {} points".format(" ",self.points))
        print("{: <30}Earned {}/{}".format(" ",self.subrounds_won,self.bet))
        print(" ")

    def displayTableBet(self, bs):
        """
        Given a bet situation (other bets, hand, points, etc.) displays information the
        human agent might need to make an informed bet decision.
        """
        # display top players hand (which will be two away in order from human player)
        #only works in 4 player games
        player_index = "Not found"
        for i,agent in enumerate(bs.agents):
            if agent.id == self.id:
                player_index = i
                break
        
        top_player_index = (player_index + 2) % 4
        right_player_index = (player_index + 3) % 4
        left_player_index = (player_index + 1) % 4
        top_player = bs.agents[top_player_index]
        left_player = bs.agents[left_player_index]
        right_player = bs.agents[right_player_index]

        
        right_player_bet = bs.other_bets[-1] if len(bs.other_bets) >= 1 else "N/A"
        top_player_bet = bs.other_bets[-2] if len(bs.other_bets) >= 2 else "N/A"
        left_player_bet = bs.other_bets[-3] if len(bs.other_bets) >= 3 else "N/A"

        print("{: <30}Player {}, {} points".format(" ",top_player.id,top_player.points))
        print("{: <30}Bet: {}".format(" ",top_player_bet))
        print(" ")
        print("{: <30}_______________________".format(" "))
        print("{: <30}|Bets:                |".format(" "))
        for line_num in range(4):
            to_print = ""
            if line_num < len(bs.other_bets):
                to_print = "|{: <21}|      ".format(bs.other_bets[line_num])
            else:
                to_print = "|{: <21}|      ".format("--")
            if line_num == 0:
                left_player_text = "{: <30}".format(f"Player {left_player.id}, {left_player.points} points")
                right_player_text = "{: <30}".format(f"Player {right_player.id}, {right_player.points} points")
                to_print = left_player_text + to_print +right_player_text
            elif line_num == 1:
                left_player_text = "{: <30}".format(f"Bet: {left_player_bet}")
                right_player_text = "{: <30}".format(f"Bet: {right_player_bet}")
                to_print = left_player_text + to_print + right_player_text
            else:
                to_print = "{: <30}".format(" ") + to_print + "{: <30}".format(" ")
            print(to_print)

        print("{: <30}_______________________       ".format(" "))
        print("")
        print("{: <30}You, {} points".format(" ",self.points))
        # print("{: <30}Earned {}/{}".format(" ",self.subrounds_won,self.bet))
        print(" ")


    def printHand(self,trump):
        """
        prints hand in comprehensible way.

        This function is specifically for human agents - labels cards either with
        a numerical index if they can be selected, or with an X if they are illegal to play.
        """
        suits = ["Spades","Hearts","Diamonds","Clubs","No Trump"]
        suits[trump] = "~"+suits[trump]+"~"
        print("{: <23}{: <23}{: <23}{: <23}".format(suits[0],suits[1],suits[2],suits[3]))
        suit_dict = {0:[],1:[],2:[],3:[]}

        for i, card in enumerate(self.hand):
            index = str(i) if card in self.available_cards else "X"
            #if a card num or X has already been added to the name, replace it
            if len(card.name.split()) > 3:
                new_card_name_list = [index + ":"]+card.name.split()[1:]
                card.name = " ".join(new_card_name_list)
            else:
                card.name = str(i)+": "+card.name
            suit_dict[card.suit].append(card)
        
        max_cards_in_suit = max([len(val) for val in suit_dict.values()])
        for row in range(max_cards_in_suit):
            for i in range(4):
                try:
                    print("{: <23}".format(suit_dict[i][row].name),end="")
                except IndexError:
                    print("{: <23}".format("--"),end="")
            print(" ")
        print(" ")
    
    def makeBet(self, bs):
        """
        Allows human to input a bet based on the situation
        """
        self.displayTableBet(bs)
        self.printHand(bs.trump)
        bets_so_far = sum(bs.other_bets)
        bet_ind = len(bs.other_bets)

        bets_left = bs.hand_size-bets_so_far
        probs = []
        possible_bets = [str(i) for i in range(bs.hand_size+1) if (i!= bets_left or bet_ind != 3)]
        bet = input(f"Input bet (valid bets are {possible_bets}):\n")
        while bet not in possible_bets:
            bet = input (f"Invalid bet - valid bets are {possible_bets}\n")
        self.bet = int(bet)
        
        return self.bet

    def chooseCard(self,srs):
        self.displayTable(srs)
        self.determineCardOptions(srs)
        self.printHand(srs.trump)
        
        while True:
            card_index = input("# Card to play:")
            if int(card_index) < len(self.hand):
                card_index = int(card_index)
                if self.hand[card_index] in self.available_cards:
                    break

            print("Try again - invalid card")
        
        card_to_play = self.hand[card_index]

        return card_to_play