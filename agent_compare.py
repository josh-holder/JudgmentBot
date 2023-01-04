from JudgmentGame import JudgmentGame
from HumanBetAgent import HumanBetAgent
from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from DQNAgent import DQNAgent
import random
from copy import deepcopy

def compareAgents(agents_to_compare,games_num=1000):
    scores = [0,0,0,0]
    for game_num in range(games_num):
        print("Simulating game {}/{}".format(game_num,games_num),end='\r')
        starting_agent = random.choice(range(len(agents_to_compare)))

        random_agent_order = agents_to_compare[starting_agent:]+agents_to_compare[:starting_agent]

        jg = JudgmentGame(agents=random_agent_order)
        resulting_agents = jg.playGame()
        for i,agent in enumerate(resulting_agents):
            scores[i] += agent.points
            agent.points = 0 #reset points

    avg_scores = [score/games_num for score in scores]
    print("Average Final Scores over {} games: {}".format(games_num,avg_scores))

    return avg_scores

if __name__ == "__main__":
    # compareAgents([DQNAgent(0),HumanBetAgent(1),SimpleAgent(2),JudgmentAgent(3)],games_num=100)
    compareAgents([DQNAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)],games_num=100)