from JudgmentGame import JudgmentGame
from HumanBetAgent import HumanBetAgent
from SimpleAgent import SimpleAgent
from JudgmentAgent import JudgmentAgent
from NNAgent import NNAgent
import random
from judgment_value_models import initActionModel, initBetModel, initEvalModel
from multiprocessing import cpu_count, Pool
from itertools import repeat
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

def compareAgentsPoolInitSlow(agents_to_compare_init):
    """
    We use an initialize function for the pool so that we can pass each worker
    the models (and go through the expensive serialization process) only once
    for each worker.
    """
    global agents_to_compare
    agents_to_compare = agents_to_compare_init

def compareAgentsPoolInit(action_model_weights, bet_model_weights, eval_model_weights):
    """
    We use an initialize function for the pool so that we can pass each worker
    the model weights (and go through the expensive serialization process) only once
    for each worker.

    Using model weights is faster than using the models themselves, though.
    """
    global agents_to_compare

    agents_to_compare = []
    for i, action_model_weight,bet_model_weight,eval_model_weight in zip(range(len(action_model_weights)), action_model_weights,bet_model_weights,eval_model_weights):
        agent = NNAgent(i, load_models=False)
        agent.action_model = initActionModel()
        agent.action_model.set_weights(action_model_weight)

        agent.bet_model = initBetModel()
        agent.bet_model.set_weights(bet_model_weight)

        agent.eval_model = initEvalModel()
        agent.eval_model.set_weights(eval_model_weight)

        agents_to_compare.append(agent)

def compareAgentsPoolSubprocess(pid):
    scores = [0,0,0,0]

    random.shuffle(agents_to_compare)
    jg = JudgmentGame(agents=agents_to_compare)
    resulting_agents = jg.playGame() #in order of agent ID
    for i,agent in enumerate(resulting_agents):
        scores[i] += agent.points
        agent.points = 0

    return scores

def compareAgents(agents_to_compare,games_num,cores=1,optimized=True):
    """
    Given a list of agents, compare them against each other in group games.

    If optimized=True, passes the action models to the pool workers as weights, not as models,
    to save a massive amount of time in serialization.

    Otherwise, passes the actual agents_to_compare structure to the pool workers, which
    is more compatible with models of different structures but is slower. 
    """
    start = time.time()
    if optimized: #Optimized but less compatible method of comparison
        action_model_weights = [agent.action_model.get_weights() for agent in agents_to_compare]
        bet_model_weights = [agent.bet_model.get_weights() for agent in agents_to_compare]
        eval_model_weights = [agent.eval_model.get_weights() for agent in agents_to_compare]
        
        initargs = [action_model_weights, bet_model_weights, eval_model_weights]
        initfcn = compareAgentsPoolInit
    else: 
        initargs = [agents_to_compare]
        initfcn = compareAgentsPoolInitSlow

    with Pool(processes=cores, initializer=initfcn, initargs=initargs) as p:
        scores = []
        #The result of imap_unordered gets filled over time, so by iterating over it
        #we can track its progress.
        for i, score in enumerate(p.imap_unordered(compareAgentsPoolSubprocess, range(games_num))):
            print(f"Simulated comparison game {i+1}/{games_num}...", end='\r')
            scores.append(score)

    print(f"Simulated {games_num} comparison games in {time.time()-start} seconds.")
    avg_scores = [sum(x)/len(x) for x in zip(*scores)]
    print(f"Average final scores: {avg_scores}")
    return scores

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    # compareAgents([NNAgent(0),HumanBetAgent(1),SimpleAgent(2),JudgmentAgent(3)],games_num=10,cores=cpu_count())
    # compareAgents([NNAgent(0),HumanBetAgent(1),HumanBetAgent(2),HumanBetAgent(3)],games_num=100)
    agents = [NNAgent(0, bet_model_name='new_test/best_bet_model', action_model_name='new_test/best_act_model', eval_model_name='new_test/best_eval_model'),\
              NNAgent(1, bet_model_name='new_test/best_bet_model', action_model_name='new_test/best_act_model', eval_model_name='new_test/best_eval_model'),\
              NNAgent(2, bet_model_name='current_best_models/best_bet_model', action_model_name='current_best_models/best_act_model', eval_model_name='current_best_models/best_eval_model'),\
                NNAgent(3, bet_model_name='current_best_models/best_bet_model', action_model_name='current_best_models/best_act_model', eval_model_name='current_best_models/best_eval_model')]
    
    agents = [HumanBetAgent(0),HumanBetAgent(1),\
              NNAgent(2, bet_model_name='current_best_models/best_bet_model', action_model_name='current_best_models/best_act_model', eval_model_name='current_best_models/best_eval_model'),\
                NNAgent(3, bet_model_name='current_best_models/best_bet_model', action_model_name='current_best_models/best_act_model', eval_model_name='current_best_models/best_eval_model')]

    compareAgents(agents, games_num=args.games, cores=cpu_count(), optimized=False)