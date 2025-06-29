from JudgmentAgent import JudgmentAgent
import random
from google import genai
import re
import os

_client = None

def get_client():
    """
    Lazily construct the Gemini client, so that agents which don't call the LLM
    can be imported without a key set.
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client

SUIT_ORDER = ["Spades", "Hearts", "Diamonds", "Clubs", "No Trump"]

def convert_card_name_to_prompt_format(card_name):
    """
    Convert card name from "2 of clubs" format to "2C" format for prompts.
    """
    # Parse the card name
    parts = card_name.split(" of ")
    if len(parts) != 2:
        return card_name  # Return as-is if not in expected format
    
    rank_part = parts[0]
    suit_part = parts[1]
    
    # Convert rank
    rank_map = {
        "Ace": "A",
        "Jack": "J",
        "Queen": "Q", 
        "King": "K",
        "10": "T"
    }
    
    if rank_part in rank_map:
        rank = rank_map[rank_part]
    else:
        rank = rank_part  # Already a number
    
    # Convert suit
    suit_map = {
        "spades": "S",
        "hearts": "H", 
        "diamonds": "D",
        "clubs": "C"
    }
    
    suit = suit_map.get(suit_part.lower(), suit_part)
    
    return f"{rank}{suit}"

"""
Agent which uses LLM calls to make decisions.
"""
class LLMAgent(JudgmentAgent):
    def __init__(self,id,llm_name):
        super().__init__(id)
        self.llm_name = llm_name

    def generateBetPrompt(self, bs):
        """
        Based on the bet situation, generate a prompt for the LLM to make a bet. Based on bet_prompt.txt.
        """
        game_specific_prompt = "'''"
        
        # Read the bet prompt template
        prompt_start = ""
        try:
            with open('bet_prompt.txt', 'r') as f:
                prompt_start = f.read()
        except FileNotFoundError:
            return "Make a bet between 0 and {}: ".format(bs.hand_size)

        # Convert hand cards to prompt format
        hand_cards = []
        self.hand.sort(key=lambda x: x.suit)
        for card in self.hand:
            hand_cards.append(convert_card_name_to_prompt_format(card.name))
        
        # Build the game state information
        trump_suit = SUIT_ORDER[bs.trump] if bs.trump < 4 else "No Trump"
        
        # Build the game state string
        game_state = f"""\n
Round: {bs.round_num}
Hand size: {bs.hand_size}
Trump suit: {trump_suit}
"""

        game_specific_prompt += game_state

        game_specific_prompt += "\n\n"

        possible_bets = list(range(bs.hand_size+1))
        illegal_bid_string = f""
        if len(bs.other_bets) == (len(bs.agents)-1):
            invalid_bet = bs.hand_size-sum(bs.other_bets)
            if invalid_bet in possible_bets:
                possible_bets.remove(invalid_bet)
                illegal_bid_string = f"invalid bid {invalid_bet}, "

        # Add information about each player
        for i, agent in enumerate(bs.agents):
            if agent.bet == -1:
                bet = "TBD"
            else:
                bet = agent.bet

            if agent.id == self.id:
                game_specific_prompt += f"You: bid TBD, {illegal_bid_string}score {self.points}, hand {{{', '.join(hand_cards)}}}\n\n"
            else:
                game_specific_prompt += f"Player {agent.id}: bid {bet}, score {agent.points}\n\n"
        
        game_specific_prompt += "'''"

        return prompt_start + game_specific_prompt

    def generatePlayPrompt(self, srs):
        """
        Based on the subround situation, generate a prompt for the LLM to play a card. Based on play_prompt.txt.
        """
        # Convert hand cards to prompt format
        hand_cards = []
        self.hand.sort(key=lambda x: x.suit)
        for card in self.hand:
            hand_cards.append(convert_card_name_to_prompt_format(card.name))

        # Convert hand cards to prompt format
        available_cards = []
        for card in self.available_cards:
            available_cards.append(convert_card_name_to_prompt_format(card.name))
        
        game_specific_prompt = "'''"
        # Read the play prompt template
        prompt_start = ""
        try:
            with open('play_prompt.txt', 'r') as f:
                prompt_start = f.read()
        except FileNotFoundError:
            return "Choose a card to play: {available_cards}"

        # Build the game state information
        trump_suit = SUIT_ORDER[srs.trump] if srs.trump < 4 else "No Trump"
        
        # Build the game state string
        game_specific_prompt += f"""\n
Round: {srs.round_num}
Hand size: {srs.hand_size}
Trump suit: {trump_suit}
"""
        
        game_specific_prompt += "\n\n"

        # Add information about each player
        for i, agent in enumerate(srs.agents):
            lead_indicator = " (Lead)" if i == 0 else ""
            
            # Determine what card was played this subround
            if i < len(srs.card_stack):
                card_played = convert_card_name_to_prompt_format(srs.card_stack[i].name)
            else:
                card_played = "N/A"
            
            # Determine which suits the agent is visibly out of
            out_of_suits = []
            for suit in range(4):
                if agent.visibly_out_of_suit[suit] == 1:
                    out_of_suits.append(SUIT_ORDER[suit][0]) #just the first letter of the suit

            prev_cards_played = [convert_card_name_to_prompt_format(card.name) for card in agent.previously_played_cards]

            if agent.id == self.id:
                game_specific_prompt += f"You: bid {agent.bet}, tricks {agent.subrounds_won}, score {agent.points}, prev cards played {prev_cards_played}, out of suits {out_of_suits}, hand {hand_cards}, legal cards {available_cards}\n\n"
            else:
                game_specific_prompt += f"Player {agent.id}{lead_indicator}: card played {card_played}, bid {agent.bet}, tricks {agent.subrounds_won}, score {agent.points}, prev cards played {prev_cards_played}, out of suits {out_of_suits}\n\n"
        
        game_specific_prompt += "'''"

        return prompt_start + game_specific_prompt

    def makeBet(self, bs):
        """
        Call LLM and return the bet.
        """
        # Generate the prompt for betting
        prompt = self.generateBetPrompt(bs)

        try:
            # Call the LLM
            response = get_client().models.generate_content(
                model=self.llm_name, 
                contents=prompt
            )

            # Extract the response text
            response_text = response.text.strip()

            # Parse the response using regex to find {number} format
            bet_match = re.search(r'\{(\d+)\}', response_text)

            if bet_match:
                bet_value = int(bet_match.group(1))
                # Validate the bet is within valid range
                if 0 <= bet_value <= bs.hand_size:
                    # Check if this bet would make the total equal to hand_size (invalid for last bidder)
                    if len(bs.other_bets) == (len(bs.agents) - 1):
                        total_bets = sum(bs.other_bets) + bet_value
                        if total_bets == bs.hand_size:
                            # Invalid bet, choose randomly from valid options
                            possible_bets = list(range(bs.hand_size + 1))
                            possible_bets.remove(bs.hand_size - sum(bs.other_bets))
                            bet_value = random.choice(possible_bets)

                    self.bet = bet_value
                    return self.bet

            # If parsing failed or bet was invalid, fall back to random choice
            possible_bets = list(range(bs.hand_size + 1))
            if len(bs.other_bets) == (len(bs.agents) - 1):
                invalid_bet = bs.hand_size - sum(bs.other_bets)
                if invalid_bet in possible_bets:
                    possible_bets.remove(invalid_bet)

            self.bet = random.choice(possible_bets)
            print(f"ERROR: LLM Agent {self.id} made a random bet: {self.bet}")
            return self.bet

        except Exception as e:
            # If LLM call fails, fall back to random choice
            possible_bets = list(range(bs.hand_size + 1))
            if len(bs.other_bets) == (len(bs.agents) - 1):
                invalid_bet = bs.hand_size - sum(bs.other_bets)
                if invalid_bet in possible_bets:
                    possible_bets.remove(invalid_bet)

            self.bet = random.choice(possible_bets)
            return self.bet

    def chooseCard(self, srs):
        """
        Call LLM and return the card.
        """
        # Generate the prompt for playing a card
        prompt = self.generatePlayPrompt(srs)

        if len(self.available_cards) == 1:
            return self.available_cards[0]
        
        try:
            # Call the LLM
            response = get_client().models.generate_content(
                model=self.llm_name, 
                contents=prompt
            )
            
            # Extract the response text
            response_text = response.text.strip()
            
            # Parse the response using regex to find {RankSuit} format
            card_match = re.search(r'\{([2-9TJQKA][SHCD])\}', response_text)
            
            if card_match:
                card_str = card_match.group(1)
                # Find the card in available cards
                for card in self.available_cards:
                    if convert_card_name_to_prompt_format(card.name) == card_str:
                        return card
            
            print(f"ERROR: LLM Agent selected invalid card, Agent {self.id} selected a random card.")
            # If parsing failed or card not found, fall back to random choice
            return random.choice(self.available_cards)
            
        except Exception as e:
            print(f"ERROR: LLM call failed, Agent {self.id} selected a random card.")
            # If LLM call fails, fall back to random choice
            return random.choice(self.available_cards)