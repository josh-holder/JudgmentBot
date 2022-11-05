from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from JudgmentGame import JudgmentGame
from NNAgent import NNAgent
from HumanAgent import HumanAgent
play_verbose = 0

if __name__ == "__main__":
    ha = HumanAgent(0)
    jg = JudgmentGame(agents=[HumanAgent(0),NNAgent(1),NNAgent(2),NNAgent(3)],print_final_tables=1)
    jg.playGame()