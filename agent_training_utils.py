import os
import tensorflow as tf
import nn_config
import time
from NNAgent import NNAgent
from compare_agents import compareAgents
from multiprocessing import cpu_count
from copy import copy
import wandb
from multiprocessing import cpu_count
import time
from copy import copy, deepcopy
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from scipy import stats
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def loadModels(args):
    """
    Given command line arguments, loads current, best, and baseline models.

    (At the start, these are all the same models - only once training begins do they diverge)
    """
    import keras
    if args.models_path == None:
        curr_bet_model_path = os.path.join(os.getcwd(),args.bet_model_path)
        curr_bet_model = keras.models.load_model(curr_bet_model_path, compile=False)
        print(f"Loaded current bet model from {curr_bet_model_path}")

        curr_eval_model_path = os.path.join(os.getcwd(),args.eval_model_path)
        curr_eval_model = keras.models.load_model(curr_eval_model_path, compile=False)
        print(f"Loaded current eval model from {curr_eval_model_path}")

        curr_action_model_path = os.path.join(os.getcwd(),args.action_model_path)
        curr_action_model = keras.models.load_model(curr_action_model_path, compile=False)
        print(f"Loaded current action model fron {curr_action_model_path}")
    
    else:
        curr_bet_model_path = os.path.join(os.getcwd(),args.models_path,"best_bet_model")
        curr_bet_model = keras.models.load_model(curr_bet_model_path, compile=False)
        print(f"Loaded current bet model from {curr_bet_model_path}")

        curr_eval_model_path = os.path.join(os.getcwd(),args.models_path,"best_eval_model")
        curr_eval_model = keras.models.load_model(curr_eval_model_path, compile=False)
        print(f"Loaded current eval model from {curr_eval_model_path}")

        curr_action_model_path = os.path.join(os.getcwd(),args.models_path,"best_act_model")
        curr_action_model = keras.models.load_model(curr_action_model_path, compile=False)
        print(f"Loaded current action model fron {curr_action_model_path}")

    best_bet_model = tf.keras.models.clone_model(curr_bet_model)
    best_bet_model.set_weights(curr_bet_model.get_weights())
    best_bet_model.compile()

    best_eval_model = tf.keras.models.clone_model(curr_eval_model)
    best_eval_model.set_weights(curr_eval_model.get_weights())
    best_eval_model.compile()

    best_action_model = tf.keras.models.clone_model(curr_action_model)
    best_action_model.set_weights(curr_action_model.get_weights())
    best_action_model.compile()

    for layer in best_bet_model.layers:
        layer.trainable = False
    for layer in best_eval_model.layers:
        layer.trainable = False
    for layer in best_action_model.layers:
        layer.trainable = False

    baseline_bet_model = tf.keras.models.clone_model(curr_bet_model)
    baseline_bet_model.set_weights(curr_bet_model.get_weights())
    baseline_bet_model.compile()

    baseline_eval_model = tf.keras.models.clone_model(curr_eval_model)
    baseline_eval_model.set_weights(curr_eval_model.get_weights())
    baseline_eval_model.compile()

    baseline_action_model = tf.keras.models.clone_model(curr_action_model)
    baseline_action_model.set_weights(curr_action_model.get_weights())
    baseline_action_model.compile()

    for layer in baseline_bet_model.layers:
        layer.trainable = False
    for layer in baseline_eval_model.layers:
        layer.trainable = False
    for layer in baseline_action_model.layers:
        layer.trainable = False

    print("Initialized baseline and best models as untrainable copies of current models.")

    return curr_bet_model, curr_eval_model, curr_action_model,\
            best_bet_model, best_eval_model, best_action_model,\
            baseline_bet_model, baseline_eval_model, baseline_action_model

def saveModels(run_name, bet_model, eval_model, act_model):
    """
    Saves the best bet, eval, and action models to the run directory.
    """
    bet_model_path = os.path.join(os.getcwd(),run_name,"best_bet_model")
    bet_model.save(bet_model_path)

    eval_model_path = os.path.join(os.getcwd(),run_name,"best_eval_model")
    eval_model.save(eval_model_path)

    act_model_path = os.path.join(os.getcwd(),run_name,"best_act_model")
    act_model.save(act_model_path)
    print("Saved bet, eval, and action models.")

def evaluateModelPerformance(curr_action_model, curr_bet_model, curr_eval_model,\
                             best_action_model, best_bet_model, best_eval_model,\
                             baseline_action_model, baseline_bet_model, baseline_eval_model,\
                             iterations_without_improving, args):
    """
    Compares agents in a rigorous way. Two copies of the current agent play in a game
    against one copy of the baseline agent, and one copy of the current best agent.

    [curr_agent1, curr_agent2, best_agent, base_agent]

    A model is determined to be a better model if there is X% confidence that curr_agent1 scores higher than best_agent
    AND that curr_agent2 scores higher than base_agent on average.

    If so, and both the current and the best agent are X% confident to beat the baseline agent by 25 points,
    then replace the baseline agent with the previous best agent.
    """
    print(f"Evaluating model performance against baseline:")
    start = time.time()
    #epsilon is zero for evaluation.
    agents_to_compare = [NNAgent(0,load_models=False),NNAgent(1,load_models=False),NNAgent(2,load_models=False),NNAgent(3,load_models=False)]
    #Initialize current agents in slots 0 and 1
    for agent in agents_to_compare[0:2]:
        agent.action_model = curr_action_model
        agent.bet_model = curr_bet_model
        agent.eval_model = curr_eval_model

        agent.action_model.compile()
        agent.bet_model.compile()
        agent.eval_model.compile()

    agents_to_compare[2].action_model = best_action_model
    agents_to_compare[2].bet_model = best_bet_model
    agents_to_compare[2].eval_model = best_eval_model
    agents_to_compare[2].action_model.compile()
    agents_to_compare[2].bet_model.compile()
    agents_to_compare[2].eval_model.compile()

    agents_to_compare[3].action_model = baseline_action_model
    agents_to_compare[3].bet_model = baseline_bet_model
    agents_to_compare[3].eval_model = baseline_eval_model
    agents_to_compare[3].action_model.compile()
    agents_to_compare[3].bet_model.compile()
    agents_to_compare[3].eval_model.compile()

    # Run half the games required for comparison and test scores:
    initial_compare_games = nn_config.COMPARISON_GAMES//2
    init_scores = compareAgents(agents_to_compare,games_num=initial_compare_games, cores=cpu_count())

    curr1_scores = [scores[0] for scores in init_scores]
    curr2_scores = [scores[1] for scores in init_scores]
    best_scores = [scores[2] for scores in init_scores]
    base_scores = [scores[3] for scores in init_scores]

    #Test scores between current and best agent
    _, chance_best_is_better = stats.ttest_rel(curr1_scores, best_scores, alternative='greater')

    #Test scores between current and baseline agent
    _, chance_base_is_better = stats.ttest_rel(curr2_scores, base_scores, alternative='greater')

    chance_better_than_best = 1-chance_best_is_better
    chance_better_than_base = 1-chance_base_is_better
    chance_better_than_both = min(chance_better_than_best, chance_better_than_base)

    if chance_better_than_both < nn_config.CONFIDENCE_REQ and chance_better_than_both > nn_config.FURTHER_EXPLORE_CONFIDENCE_REQ:
        print(f"~~~{nn_config.CONFIDENCE_REQ}>{chance_better_than_both}>{nn_config.FURTHER_EXPLORE_CONFIDENCE_REQ} chance that the model is better than both baseline and best, so simulate a few more games.~~~")

        #Run the rest of the games required for comparison and test scores:
        rest_compare_games = nn_config.COMPARISON_GAMES - initial_compare_games
        rest_scores = compareAgents(agents_to_compare,games_num=rest_compare_games, cores=cpu_count())

        total_scores = init_scores + rest_scores

        curr1_scores = [scores[0] for scores in total_scores]
        curr2_scores = [scores[1] for scores in total_scores]
        best_scores = [scores[2] for scores in total_scores]
        base_scores = [scores[3] for scores in total_scores]

        #Test scores between current and best agent
        _, chance_best_is_better = stats.ttest_rel(curr1_scores, best_scores, alternative='greater')

        #Test scores between current and baseline agent
        _, chance_base_is_better = stats.ttest_rel(curr2_scores, base_scores, alternative='greater')

        chance_better_than_best = 1-chance_best_is_better
        chance_better_than_base = 1-chance_base_is_better
        chance_better_than_both = min(chance_better_than_best, chance_better_than_base)

    #After potentially running more games, check if the model is better than both baseline and best
    if chance_better_than_both > nn_config.CONFIDENCE_REQ:
        print(f"{chance_better_than_both}>{nn_config.CONFIDENCE_REQ} chance that the model is better than both baseline and best, so save it!!")

        increased_base_scores = [base_score+25 for base_score in base_scores]
        curr_avg_scores = [(curr1+curr2)/2 for curr1, curr2 in zip(curr1_scores, curr2_scores)]
        _, chance_curr_way_better_than_base = stats.ttest_rel(curr_avg_scores, increased_base_scores, alternative='less')
        _, chance_best_way_better_than_base = stats.ttest_rel(best_scores, increased_base_scores, alternative='less')

        if chance_curr_way_better_than_base > nn_config.CONFIDENCE_REQ and chance_best_way_better_than_base > nn_config.CONFIDENCE_REQ:
            print(f"Additionally, both best and current model are >25 points better than the baseline, so change the baseline to the previous best model.")
            baseline_action_model.set_weights(best_action_model.get_weights())
            baseline_bet_model.set_weights(best_bet_model.get_weights())
            baseline_eval_model.set_weights(best_eval_model.get_weights())

        best_action_model.set_weights(curr_action_model.get_weights())
        best_bet_model.set_weights(curr_bet_model.get_weights())
        best_eval_model.set_weights(curr_eval_model.get_weights())

        saveModels(args.run_name, curr_bet_model, curr_eval_model, curr_action_model)

        iterations_without_improving = 0
    else:
        print(f"Model is not clearly better - {chance_better_than_best}, {chance_better_than_base} chance of being better than best and baseline respectively. Incrementing iterations_without_improving.")
        iterations_without_improving += 1


    if iterations_without_improving >= nn_config.ITER_WOUT_IMPROVE_BEFORE_RESET:
        print(f"It has been {iterations_without_improving} iterations without improving on best agent and baseline agent, so reset to old best agent.")

        curr_action_model.set_weights(best_action_model.get_weights())
        curr_bet_model.set_weights(best_bet_model.get_weights())
        curr_eval_model.set_weights(best_eval_model.get_weights())

        iterations_without_improving = 0

    if args.track: wandb.log({"eval/chance_better_than_best": chance_better_than_best,\
                              "eval/chance_better_than_base": chance_better_than_base,\
                              "eval/iterations_without_improving": iterations_without_improving})

    print(f"Evaluated model performance against baseline in {time.time()-start} seconds.")

    return curr_action_model, curr_bet_model, curr_eval_model, \
            best_action_model, best_bet_model, best_eval_model, \
            baseline_action_model, baseline_bet_model, baseline_eval_model, \
            iterations_without_improving

if __name__ == "__main__":
    a = NNAgent(0,bet_model_name="current_best_models/best_bet_model",action_model_name="current_best_models/best_act_model",eval_model_name="current_best_models/best_eval_model")
    start = time.time()
    pickle.dumps(a.action_model)
    pickle.dumps(a.bet_model)
    pickle.dumps(a.eval_model)
    print(time.time()-start)
    
    ba = a.action_model.get_weights()
    bb = a.bet_model.get_weights()
    be = a.eval_model.get_weights()
    start = time.time()
    pickle.dumps(ba)
    pickle.dumps(bb)
    pickle.dumps(be)
    print(time.time()-start)