from JudgmentGame import JudgmentGame
from HumanBetAgent import HumanBetAgent
from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from NNAgent import NNAgent
import random
from multiprocessing import Process, cpu_count, Queue
import time
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _build_parser():
    parser = argparse.ArgumentParser(description='Compare Judgement Agents.')

    parser.add_argument(
        '-g','--games',
        help="Number of games to compare for",
        type=int,
        default=1000,
    )

    return parser

def compareAgentsSubprocess(agents_to_compare, games_num, total_scores_queue, id):
    scores = [0,0,0,0]
    for game_num in range(games_num):
        print(f"Simulating game {game_num+1}/{games_num} for process id {id}",end='\r')

        #randomly shuffle agents so that starting agent is random
        random.shuffle(agents_to_compare)

        jg = JudgmentGame(agents=agents_to_compare)
        resulting_agents = jg.playGame() #in order of agent ID
        for i,agent in enumerate(resulting_agents):
            scores[i] += agent.points
            agent.points = 0 #reset points

    #Adding scores to the queue
    total_scores = total_scores_queue.get() #retrieve total score object
    total_scores = [total_scores[i]+scores[i] for i in range(len(scores))] 
    total_scores_queue.put(total_scores) #put total score object back into queue, with scores added

def compareAgents(agents_to_compare,games_num,cores=1):
    start = time.time()
    total_scores_queue = Queue()
    total_scores_queue.put([0,0,0,0])

    games_num = games_num + (cores - games_num % cores) #Ensure that games_num is an even multiple of the number of cores

    processes = []
    for process_num in range(cores):
        # copy_of_agent_list = deepcopy(agents_to_compare)
        p = Process(target=compareAgentsSubprocess, args=(agents_to_compare, games_num//cores, total_scores_queue, process_num))
        #Weird adam optimizer warning originates in the p.start() call below
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_scores = total_scores_queue.get()
    avg_scores = [score/games_num for score in total_scores]

    print(f"Generated {games_num} games in {time.time()-start} seconds.")
    print("Average Final Scores over {} games: {}".format(games_num,avg_scores))

    return avg_scores

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    # compareAgents([NNAgent(0),HumanBetAgent(1),SimpleAgent(2),JudgmentAgent(3)],games_num=10,cores=cpu_count())
    # compareAgents([NNAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)],games_num=100)
    compareAgents([NNAgent(0,bet_model_name="lab_run/best_bet_model",action_model_name="lab_run/best_act_model",eval_model_name="lab_run/best_eval_model"),\
            NNAgent(1,bet_model_name="lab_run/best_bet_model",action_model_name="lab_run/best_act_model",eval_model_name="lab_run/best_eval_model"),\
            NNAgent(2,bet_model_name="current_best_models/best_bet_model",action_model_name="current_best_models/best_act_model",eval_model_name="current_best_models/best_eval_model"),\
            NNAgent(3,bet_model_name="current_best_models/best_bet_model",action_model_name="current_best_models/best_act_model",eval_model_name="current_best_models/best_eval_model")], games_num=args.games, cores=cpu_count())