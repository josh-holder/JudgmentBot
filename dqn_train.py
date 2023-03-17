from collections import deque
import pickle
import os
import nn_config
import argparse
import numpy as np
import random
import keras
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from copy import copy, deepcopy
from JudgmentUtils import postProcessTrainData, postProcessBetTrainData, convertSubroundSituationToActionState
from JudgmentValueModels import initActionModel, initBetModel, initEvalModel 
from JudgmentGame import JudgmentGame
from DQNAgent import DQNAgent
from compare_agents import compareAgents
from HumanBetAgent import HumanBetAgent
from multiprocessing import cpu_count
import wandb

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def _build_parser():
    parser = argparse.ArgumentParser(description='Run NN to generate ideal playing strategy for SPLT.')

    parser.add_argument(
        '-r','--run_name',
        help="Will name the output directory for the results of the run",
        type=str,
        default="test_run",
    )

    parser.add_argument(
        '-b','--bet_model_path',
        help="Path from current directory to folder containing bet model",
        type=str,
        default="bet_expert_train_model",
    )

    parser.add_argument(
        '-e','--eval_model_path',
        help="Path from current directory to folder containing eval model",
        type=str,
        default="eval_expert_train_model",
    )

    parser.add_argument(
        '-a','--action_model_path',
        help="Path from current directory to folder containing action model",
        type=str,
        default="action_expert_train_model",
    )

    parser.add_argument(
        '-t','--track',
        help="Flag determining whether to track this run in weights and biases",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        '--wandb_project_name',
        help="Name of WandB project to log to. Only relevant if args.track=True.",
        type=str,
        default="JudgementBot",
    )

    parser.add_argument(
        '--rb_data_folder',
        help="Folder to use existing experience data from. Defaults to the run name.",
        type=str,
        default="",
    )

    parser.add_argument(
        "--wandb-entity", 
        type=str, 
        default="josh-holder", 
        help="the entity (team) of wandb's project"
    )

    return parser

def initWandBTrack(args):
    #initialize weights and biases tracking
    run_name = f"{args.run_name}__{int(time.time())}"
    config_dict = vars(args)
    config_dict.pop("track",None)
    config_dict.pop("wandb_project_name",None)
    config_dict.pop("wandb_entity",None)
    
    nn_config_dict = nn_config.__dict__
    for key in list(nn_config.__dict__.keys()):
        if key.startswith("_"): nn_config_dict.pop(key,None)

    config_dict = {**config_dict, **nn_config.__dict__}

    wandb.init(
        name=run_name,
        entity=args.wandb_entity,
        project=args.wandb_project_name,
        config=config_dict,
    )

def loadReplayBufferData(run_name, folder_name='dqn_experience_data'):
    """
    Loads exists, or creates new experience data to use for experience replay.
    """
    run_folder_path = os.path.join(os.getcwd(),run_name)
    dqnexp_folder_path = os.path.join(run_folder_path,folder_name)
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)
    if not os.path.exists(dqnexp_folder_path):
        os.mkdir(dqnexp_folder_path)

    act_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"act_experience_data.pkl")
    if os.path.exists(act_mem_path):
        print("Loading existing action experience data.")
        with open(act_mem_path,'rb') as f:
            act_experience_data = pickle.load(f)
        print(f"Done loading {len(act_experience_data)} items of action experience data")
    else:
        print("Previous action experience data not found: generating empty memory list.")
        act_experience_data = deque(maxlen=nn_config.ACTION_REPLAY_BUFFER_SIZE)

    bet_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"bet_experience_data.pkl")
    if os.path.exists(bet_mem_path):
        print("Loading existing bet experience data.")
        with open(bet_mem_path,'rb') as f:
            bet_experience_data = pickle.load(f)
        print(f"Done loading {len(bet_experience_data)} items of bet experience data")
    else:
        print("Previous bet experience data not found: generating empty memory list.")
        bet_experience_data = deque(maxlen=nn_config.BET_REPLAY_BUFFER_SIZE)

    eval_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"eval_experience_data.pkl")
    if os.path.exists(eval_mem_path):
        print("Loading existing eval experience data.")
        with open(eval_mem_path,'rb') as f:
            eval_experience_data = pickle.load(f)
        print(f"Done loading {len(eval_experience_data)} items of eval experience data")
    else:
        print("Previous eval experience data not found: generating empty memory list.")
        eval_experience_data = deque(maxlen=nn_config.EVAL_REPLAY_BUFFER_SIZE)

    return bet_experience_data, eval_experience_data, act_experience_data

def saveExperienceData(run_name,bet_rb_data,eval_rb_data,action_rb_data,folder_name="dqn_experience_data"):
    folder_path = os.path.join(os.getcwd(), run_name, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    bet_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"bet_experience_data.pkl")
    with open(bet_mem_path,'wb') as f:
        print(f"Saving {len(bet_rb_data)} items of bet experience data in {bet_mem_path}")
        pickle.dump(bet_rb_data,f)
    
    eval_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"eval_experience_data.pkl")
    with open(eval_mem_path,'wb') as f:
        print(f"Saving {len(eval_rb_data)} items of eval experience data in {eval_mem_path}")
        pickle.dump(eval_rb_data,f)
    
    act_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"act_experience_data.pkl")
    with open(act_mem_path,'wb') as f:
        print(f"Saving {len(action_rb_data)} items of state transition data in {act_mem_path}")
        pickle.dump(action_rb_data,f)

def saveModels(run_name, bet_model, eval_model, act_model):
    bet_model_path = os.path.join(os.getcwd(),run_name,"best_bet_model")
    bet_model.save(bet_model_path)

    eval_model_path = os.path.join(os.getcwd(),run_name,"best_eval_model")
    eval_model.save(eval_model_path)

    act_model_path = os.path.join(os.getcwd(),run_name,"best_act_model")
    act_model.save(act_model_path)
    print("Saved bet, eval, and action models.")

def convertMiniBatchToStateRewardPair(state_transition_minibatch, action_model, eval_model):
    """
    Convert minibatch of state transitions of the form 
    [(init_srs, card, reward),...,(init_srs, card, reward)]

    to the form:

    [([action_state],reward),...,([action_state],reward)]

    (Later, we can use postProcessTrainData() to convert this to training form)

    (NOTE: 1/27/23 changed to use rewards directly instead of transitions for lower bias from using direct Monte Carlo rewards,
    lower storage needs, lower computation requirements, etc. Form used to be:
    [(init_srs, card, final_srs),...,(init_srs, card, final_srs)])
    """
    converted_minibatch = []
    for state_transition in state_transition_minibatch:
        srs_before = state_transition[0]
        card = state_transition[1]
        reward = state_transition[2]

        #Determine current agent, set evaluation models
        curr_agent = deepcopy(srs_before.agents[len(srs_before.card_stack)])
        curr_agent.eval_model = eval_model

        action_state = convertSubroundSituationToActionState(srs_before,curr_agent,card)

        converted_minibatch.append((action_state, reward))

    return converted_minibatch
            

def trainDQNAgent():
    parser = _build_parser()
    args = parser.parse_args()
    if args.rb_data_folder == "": args.rb_data_folder = args.run_name #if no experience data was input, attempt to find experience data in current run folder


    run_folder_path = os.path.join(os.getcwd(),args.run_name)
    if not os.path.exists(run_folder_path):
        os.mkdir(run_folder_path)

    if args.track: initWandBTrack(args)

    bet_rb_data, eval_rb_data, action_rb_data = loadReplayBufferData(args.rb_data_folder)

    jg = JudgmentGame(agents=[DQNAgent(0,epsilon=0.3, load_models=False),DQNAgent(1,epsilon=0.3, load_models=False),DQNAgent(2,epsilon=0.3, load_models=False),DQNAgent(3,epsilon=0.3, load_models=False)])

    print("Loading models...")
    curr_bet_model_path = os.path.join(os.getcwd(),args.bet_model_path)
    curr_bet_model = keras.models.load_model(curr_bet_model_path)
    baseline_bet_model = keras.models.load_model(curr_bet_model_path)
    print(f"Loaded bet model from {curr_bet_model_path}")

    curr_eval_model_path = os.path.join(os.getcwd(),args.eval_model_path)
    curr_eval_model = keras.models.load_model(curr_eval_model_path)
    baseline_eval_model = keras.models.load_model(curr_eval_model_path)
    print(f"Loaded eval model from {curr_eval_model_path}")

    curr_action_model_path = os.path.join(os.getcwd(),args.action_model_path)
    curr_action_model = keras.models.load_model(curr_action_model_path)
    baseline_action_model = keras.models.load_model(curr_action_model_path)
    print(f"Loaded action model fron {curr_action_model_path}")

    for agent in jg.agents:
        agent.action_model = curr_action_model
        agent.bet_model = curr_bet_model
        agent.eval_model = curr_eval_model

    #Wait until all experience banks are at least 1/4 full to start learning:
    # while len(action_rb_data) < 250:
    print("Generating initial amount of training data...")
    need_to_generate_init_data = False
    while len(bet_rb_data)<nn_config.BET_REPLAY_BUFFER_SIZE/4 or len(eval_rb_data)<nn_config.EVAL_REPLAY_BUFFER_SIZE/4 \
        or len(action_rb_data)<nn_config.ACTION_REPLAY_BUFFER_SIZE/4:
        start = time.time()
        need_to_generate_init_data = True
        bet_data, eval_data, state_transitions = jg.playGameAndCollectData(use_in_replay_buffer=True)
        print(time.time()-start)

        #add to existing bet_rb_data bank
        bet_rb_data.extend(bet_data)
        eval_rb_data.extend(eval_data)
        action_rb_data.extend(state_transitions)

        print(f"Bet: {len(bet_rb_data)}/{nn_config.BET_REPLAY_BUFFER_SIZE/4}, Eval: {len(eval_rb_data)}/{nn_config.EVAL_REPLAY_BUFFER_SIZE/4}, Act: {len(action_rb_data)}/{nn_config.ACTION_REPLAY_BUFFER_SIZE/4}",end='\r')

        jg.resetGame()

    print(f"Sufficient training data is available ({len(action_rb_data)} state transition, {len(eval_rb_data)} eval, {len(bet_rb_data)} bet)")
    if need_to_generate_init_data:
        saveExperienceData(args.run_name, bet_rb_data, eval_rb_data, action_rb_data)
        # #OPTIONAL: also save in standard_bet_expert_exp_data
        # saveExperienceData("agent_1_28_init_rbuffer_data", bet_rb_data, eval_rb_data, action_rb_data, folder_name="")

    performance_against_best_agents = []
    new_states_to_achieve_performances = []

    new_state_action_pairs_trained_on = 0

    best_bet_weights = copy(curr_bet_model.get_weights())
    best_action_weights = copy(curr_action_model.get_weights())
    best_eval_weights = copy(curr_eval_model.get_weights())

    best_model_beat_baseline_by = 0 #track how much the previous agent beat the baseline agent by to determine if the nnew agent is better
    iterations_without_improving_best_agent = 0

    while True:
        new_state_transitions = 0

        num_new_transitions_before_eval_bet_training = nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN
        act_model_train_start = time.time()
        print(f"Playing games and training action model for {num_new_transitions_before_eval_bet_training} state transitions")
        while new_state_transitions < num_new_transitions_before_eval_bet_training:
            bet_data, eval_data, state_transitions = jg.playGameAndCollectData(use_in_replay_buffer=True)

            new_state_transitions += len(state_transitions)
            print(f"State transitions: {new_state_transitions}/{num_new_transitions_before_eval_bet_training}",end='\r')

            #add to existing bet_rb_data bank
            bet_rb_data.extend(bet_data)
            eval_rb_data.extend(eval_data)
            action_rb_data.extend(state_transitions)

            minibatch = random.sample(action_rb_data,min(nn_config.RETRAIN_BATCH_SIZE, len(action_rb_data))) #selects BATCH_SIZE unique states to retrain the NN on
            new_state_action_pairs_trained_on += min(nn_config.RETRAIN_BATCH_SIZE, len(action_rb_data))

            #Convert minibatch from (init_srs, action, reward) to (action_state, reward)
            minibatch = convertMiniBatchToStateRewardPair(minibatch, curr_action_model, curr_eval_model)
            minibatch_inputs = []
            minibatch_outputs = []

            #Convert minibatch to correct form for inputs and outputs to train model on
            for minibatch_input_output in minibatch:
                minibatch_inputs.append(minibatch_input_output[0])
                minibatch_outputs.append(minibatch_input_output[1])
            minibatch_inputs = postProcessTrainData(minibatch_inputs)
            minibatch_outputs = np.array(minibatch_outputs)

            weights_before_fit = copy(curr_action_model.get_weights())

            act_loss = curr_action_model.fit(minibatch_inputs, minibatch_outputs, epochs=nn_config.RETRAIN_EPOCHS, batch_size=nn_config.RETRAIN_BATCH_SIZE, verbose=0)

            wandb.log({"act_train/state_transitions":new_state_action_pairs_trained_on,"act_train/act_loss":act_loss.history["loss"][-1]})

            #Ensure that the model weights did not blow up
            new_action_model_has_nan = False
            for weight_layers in curr_action_model.get_weights():
                for weight_layer in weight_layers:
                    if np.any(np.isnan(weight_layer)):
                        print("Action model blew up and produced NaN weights - resetting to before.")
                        print(f"Old weights: {curr_action_model.get_weights()}")
                        curr_action_model.set_weights(weights_before_fit)
                        print(f"New weights: {curr_action_model.get_weights()}")

                        new_action_model_has_nan = True
                        break
                if new_action_model_has_nan == True: break

            jg.resetGame()

            #Set action models of agents to the new action model
            for agent in jg.agents:
                agent.action_model = curr_action_model

        saveExperienceData(args.run_name, bet_rb_data, eval_rb_data, action_rb_data)

        #~~~~~~~~~~~~~~~~~~~~~~~TRAINING BET AND EVAL NETWORKS ON NEW EXPERIENCE DATA~~~~~~~~~~~~~~~~~~~``
        print(f">{num_new_transitions_before_eval_bet_training} new transitions generated in {time.time()-act_model_train_start} sec, so retraining bet and evaluation networks on new data.")
        bet_eval_train_start = time.time()

        bet_data_inputs = []
        bet_data_outputs = []
        for bet_data in bet_rb_data:
            bet_data_inputs.append(bet_data[0])
            bet_data_outputs.append(bet_data[1])

        bet_data_inputs = postProcessBetTrainData(bet_data_inputs)
        bet_data_outputs = np.array(bet_data_outputs)
        
        print("Retraining bet network...")
        bet_loss = curr_bet_model.fit(bet_data_inputs,bet_data_outputs,epochs=nn_config.BET_TRAIN_EPOCHS,batch_size=nn_config.BET_TRAIN_BATCH_SIZE,verbose=0)
        print("Done retraining bet network.")

        eval_data_inputs = []
        eval_data_outputs = []
        for eval_data in eval_rb_data:
            eval_data_inputs.append(eval_data[0])
            eval_data_outputs.append(eval_data[1])

        eval_data_inputs = postProcessTrainData(eval_data_inputs)
        eval_data_outputs = np.array(eval_data_outputs)

        print("Retraining eval network...")
        eval_loss = curr_eval_model.fit(eval_data_inputs,eval_data_outputs,epochs=nn_config.EVAL_TRAIN_EPOCHS,batch_size=nn_config.EVAL_TRAIN_BATCH_SIZE,verbose=0)

        print(f"Done retraining bet and eval network in {time.time()-bet_eval_train_start} seconds.")

        #Updating action and evaluation models for agents
        for agent in jg.agents:
            agent.bet_model = curr_bet_model
            agent.eval_model = curr_eval_model

        wandb.log({"be_train/bet_loss":bet_loss.history["loss"][-1],"be_train/eval_loss":eval_loss.history["loss"][-1]})

        #~~~~~~~~~~~~~~~~~~~~~~EVALUATING PERFORMANCE~~~~~~~~~~~~~~~~~~~~~~~
        print("Bet and Evaluation models retrained on new data: evaluating performance.")

        #epsilon is zero for evaluation. Load new models into agents 0 and 1, best models into agents 2 and 3.
        agents_to_compare = [DQNAgent(0,load_models=False),DQNAgent(1,load_models=False),DQNAgent(2,load_models=False),DQNAgent(3,load_models=False)]
        for agent in agents_to_compare[0:2]:
            agent.action_model = curr_action_model
            agent.bet_model = curr_bet_model
            agent.eval_model = curr_eval_model

        for agent in agents_to_compare[2:]:
            agent.action_model = baseline_action_model
            agent.bet_model = baseline_bet_model
            agent.eval_model = baseline_eval_model

        print("Performance against baseline agent:")
        avg_scores_against_baseline_agents = compareAgents(agents_to_compare,games_num=24, cores=cpu_count())
        new_agent_score_against_baseline = sum(avg_scores_against_baseline_agents[0:2])/len(avg_scores_against_baseline_agents[0:2])
        baseline_agent_score_against_new = sum(avg_scores_against_baseline_agents[2:])/len(avg_scores_against_baseline_agents[2:])
        print(f"New Agent average score: {new_agent_score_against_baseline}, baseline agent average score: {baseline_agent_score_against_new}")

        current_beat_baseline_by = new_agent_score_against_baseline - baseline_agent_score_against_new

        # print("Performance against HumanBetAgents:")
        # avg_scores_against_humanbet_agents = compareAgents([agents_to_compare[0],agents_to_compare[1],HumanBetAgent(2),HumanBetAgent(3)],games_num=20, cores=cpu_count())
        # new_agent_score_against_humanbet = sum(avg_scores_against_humanbet_agents[0:2])/len(avg_scores_against_humanbet_agents[0:2])
        # humanbet_agent_score_against_new = sum(avg_scores_against_humanbet_agents[2:])/len(avg_scores_against_humanbet_agents[2:])
        # print(f"New Agent average score: {new_agent_score_against_humanbet}, HumanBet agent average score: {humanbet_agent_score_against_new}")

        if (current_beat_baseline_by > best_model_beat_baseline_by):
            print(f"!!!New agent improves on current best agent (beating baseline by {current_beat_baseline_by} instead of {best_model_beat_baseline_by}), so saving it.!!!")
            best_bet_weights = copy(curr_bet_model.get_weights())
            best_action_weights = copy(curr_action_model.get_weights())
            best_eval_weights = copy(curr_eval_model.get_weights())

            saveModels(args.run_name, curr_bet_model, curr_eval_model, curr_action_model)

            iterations_without_improving_best_agent = 0
            best_model_beat_baseline_by = current_beat_baseline_by
        else:
            iterations_without_improving_best_agent += 1
            
            if iterations_without_improving_best_agent >= 3:
                print(f"It has been {iterations_without_improving_best_agent} iterations without improving on best agent, so reset to old best agent.")
                #remove the last ~50 games of data from the buffer
                action_rb_data = [action_rb_data.pop() for _i in range(3*nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN)]
                eval_rb_data = [eval_rb_data.pop() for _i in range(3*nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN//3)] 
                bet_rb_data = [bet_rb_data.pop() for _i in range(3*nn_config.NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN//6)]

                curr_bet_model.set_weights(best_bet_weights)
                curr_action_model.set_weights(best_action_weights)
                curr_eval_model.set_weights(best_eval_weights)

                iterations_without_improving_best_agent = 0
            else: print(f"~~~New agent does not improve on best agent (beat baseline by {current_beat_baseline_by} instead of {best_model_beat_baseline_by}), so increase iterations without improving on best to {iterations_without_improving_best_agent} and continue training.~~~")

        print(f"Evaluation complete: generating {num_new_transitions_before_eval_bet_training} new state transitions.")

        wandb.log({"eval/score_diff_against_baseline": current_beat_baseline_by, "eval/state_transitions": new_state_action_pairs_trained_on})

        #save data for progress plots
        new_states_to_achieve_performances.append(new_state_action_pairs_trained_on)
        performance_against_best_agents.append(current_beat_baseline_by)
        plt.plot(new_states_to_achieve_performances,performance_against_best_agents)
        plt.xlabel("Number of new training examples")
        baseline_model_name = args.action_model_path.split("/")[0]
        plt.ylabel(f"Avg. Score Diff vs. Baseline Model ({baseline_model_name})")

        progress_graph_path = os.path.join(os.getcwd(), args.run_name, "performance_vs_baseline_agent_over_time.png")
        plt.savefig(progress_graph_path)
        plt.clf()

if __name__ == "__main__":
    # jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])
    # bet_rb_data, eval_rb_data, action_rb_data = loadReplayBufferData("run1")

    trainDQNAgent()

    # bet_data, eval_data, state_transitions = jg.playGameAndTrackStateTransitions()

    # for state_transition in state_transitions:
    #     start_srs = state_transition[0]
    #     action = state_transition[1]
    #     final_srs = state_transition[2]

    #     order_position = len(start_srs.card_stack)
    #     print(order_position)
    #     curr_agent = start_srs.agents[order_position]
    #     print(f"Start hand:")
    #     for card in curr_agent.hand:
    #         print(card.name)

    #     print(f"Playing card: {action.name}")
    #     if type(final_srs) != type(0.0):
    #         order_position = len(final_srs.card_stack)
    #         curr_agent = final_srs.agents[order_position]
    #         print(f"Final hand:")
    #         for card in curr_agent.hand:
    #             print(card.name)
    #     else:
    #         print(f"Round over. Reward = {final_srs}")

    #     print("~~~~~~~~~")

    # for bet_dat in bet_data:
    #     print(bet_dat[0][59],bet_dat[1])