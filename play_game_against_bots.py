from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from JudgmentGame import JudgmentGame
from HumanBetAgent import HumanBetAgent
from HumanAgent import HumanAgent
play_verbose = 0

if __name__ == "__main__":
    ha = HumanAgent(0)
    jg = JudgmentGame(agents=[HumanAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)],print_final_tables=1)
    jg.playGame()