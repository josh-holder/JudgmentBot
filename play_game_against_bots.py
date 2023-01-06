from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from JudgmentGame import JudgmentGame
from HumanBetAgent import HumanBetAgent
from HumanAgent import HumanAgent
from DQNAgent import DQNAgent
play_verbose = 0

if __name__ == "__main__":
    jg = JudgmentGame(agents=[HumanAgent(0),DQNAgent(1,bet_model_name="run1/best_bet_model",action_model_name="run1/best_act_model",eval_model_name="run1/best_eval_model"),\
            DQNAgent(2,bet_model_name="run1/best_bet_model",action_model_name="run1/best_act_model",eval_model_name="run1/best_eval_model"),\
            DQNAgent(3,bet_model_name="run1/best_bet_model",action_model_name="run1/best_act_model",eval_model_name="run1/best_eval_model")],print_final_tables=1)
    jg.playGame()