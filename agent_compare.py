from audioop import avg
from JudgmentGame import JudgmentGame
from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
import random
from copy import deepcopy

def compareAgents(agents_to_compare,games_num=1000):
    scores = [0,0,0,0]
    for game_num in range(games_num):
        print("Simulating game {}/{}".format(game_num,games_num),end='\r')
        starting_agent = random.choice(range(len(agents_to_compare)))
        random_agent_order = deepcopy(agents_to_compare[starting_agent:]+agents_to_compare[:starting_agent])
        jg = JudgmentGame(agents=random_agent_order)
        resulting_agents = jg.playGame()
        for i,agent in enumerate(resulting_agents):
            scores[i] += agent.points

    avg_scores = [score/games_num for score in scores]
    print("Average Final Scores over {} games: {}".format(games_num,avg_scores))

    return avg_scores

if __name__ == "__main__":
    compareAgents([JudgmentAgent(0),SimpleAgent(1),JudgmentAgent(2),JudgmentAgent(3)])
