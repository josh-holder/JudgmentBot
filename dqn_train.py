from collections import deque
import pickle
import os
import nn_config
import argparse
import numpy as np
import random
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from copy import copy, deepcopy
from judgment_data_utils import postProcessTrainData, postProcessBetTrainData, convertSubroundSituationToActionState
from judgment_value_models import initActionModel, initBetModel, initEvalModel
from agent_training_utils import loadModels, saveModels, evaluateModelPerformance
from JudgmentGame import JudgmentGame
from NNAgent import NNAgent
from compare_agents import compareAgents
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
        '-m','--models_path',
        help="Path to folder containing models, if you're restarting from a previous run. Defaults to None, in which case will use -b, -e, and -a to find models.",
        default=None,
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
    config_dict = vars(copy(args))
    config_dict.pop("track",None)
    config_dict.pop("wandb_project_name",None)
    config_dict.pop("wandb_entity",None)
    config_dict["algo"] = "dqn"
    
    nn_config_dict = nn_config.__dict__
    for key in list(nn_config.__dict__.keys()):
        if key.startswith("_"): nn_config_dict.pop(key,None)
        if key.startswith("A3C"): nn_config_dict.pop(key,None)

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
        act_experience_data = deque(maxlen=nn_config.DQN_ACTION_REPLAY_BUFFER_SIZE)

    bet_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"bet_experience_data.pkl")
    if os.path.exists(bet_mem_path):
        print("Loading existing bet experience data.")
        with open(bet_mem_path,'rb') as f:
            bet_experience_data = pickle.load(f)
        print(f"Done loading {len(bet_experience_data)} items of bet experience data")
    else:
        print("Previous bet experience data not found: generating empty memory list.")
        bet_experience_data = deque(maxlen=nn_config.DQN_BET_REPLAY_BUFFER_SIZE)

    eval_mem_path = os.path.join(os.getcwd(),run_name,folder_name,"eval_experience_data.pkl")
    if os.path.exists(eval_mem_path):
        print("Loading existing eval experience data.")
        with open(eval_mem_path,'rb') as f:
            eval_experience_data = pickle.load(f)
        print(f"Done loading {len(eval_experience_data)} items of eval experience data")
    else:
        print("Previous eval experience data not found: generating empty memory list.")
        eval_experience_data = deque(maxlen=nn_config.DQN_EVAL_REPLAY_BUFFER_SIZE)

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

    jg = JudgmentGame(agents=[NNAgent(0,epsilon=0.3, load_models=False),NNAgent(1,epsilon=0.3, load_models=False),NNAgent(2,epsilon=0.3, load_models=False),NNAgent(3,epsilon=0.3, load_models=False)])

    print("Loading models...")
    curr_bet_model, curr_eval_model, curr_action_model,\
        best_bet_model, best_eval_model, best_action_model,\
            baseline_bet_model, baseline_eval_model, baseline_action_model = loadModels(args)

    for agent in jg.agents:
        agent.action_model = curr_action_model
        agent.bet_model = curr_bet_model
        agent.eval_model = curr_eval_model

    #Wait until all experience banks are at least 1/4 full to start learning:
    # while len(action_rb_data) < 250:
    print("Generating initial amount of training data...")
    need_to_generate_init_data = False
    while len(bet_rb_data)<nn_config.DQN_BET_REPLAY_BUFFER_SIZE/4 or len(eval_rb_data)<nn_config.DQN_EVAL_REPLAY_BUFFER_SIZE/4 \
        or len(action_rb_data)<nn_config.DQN_ACTION_REPLAY_BUFFER_SIZE/4:
        start = time.time()
        need_to_generate_init_data = True
        bet_data, eval_data, state_transitions = jg.playGameAndCollectData(use_in_replay_buffer=True)
        print(time.time()-start)

        #add to existing bet_rb_data bank
        bet_rb_data.extend(bet_data)
        eval_rb_data.extend(eval_data)
        action_rb_data.extend(state_transitions)

        print(f"Bet: {len(bet_rb_data)}/{nn_config.DQN_BET_REPLAY_BUFFER_SIZE/4}, Eval: {len(eval_rb_data)}/{nn_config.DQN_EVAL_REPLAY_BUFFER_SIZE/4}, Act: {len(action_rb_data)}/{nn_config.DQN_ACTION_REPLAY_BUFFER_SIZE/4}",end='\r')

        jg.resetGame()

    print(f"Sufficient training data is available ({len(action_rb_data)} state transition, {len(eval_rb_data)} eval, {len(bet_rb_data)} bet)")
    if need_to_generate_init_data:
        saveExperienceData(args.run_name, bet_rb_data, eval_rb_data, action_rb_data)
        # #OPTIONAL: also save in standard_bet_expert_exp_data
        # saveExperienceData("agent_1_28_init_rbuffer_data", bet_rb_data, eval_rb_data, action_rb_data, folder_name="")

    performance_against_best_agents = []
    new_states_to_achieve_performances = []

    new_state_action_pairs_trained_on = 0

    iterations_without_improving = 0

    while True:
        new_state_transitions = 0

        num_new_transitions_before_eval_bet_training = nn_config.DQN_NUM_NEW_TRANSITIONS_BEFORE_EVAL_BET_TRAIN
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

            if args.track:
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

        if args.track: wandb.log({"be_train/bet_loss":bet_loss.history["loss"][-1],"be_train/eval_loss":eval_loss.history["loss"][-1]})

        #~~~~~~~~~~~~~~~~~~~~~~EVALUATING PERFORMANCE~~~~~~~~~~~~~~~~~~~~~~~
        print("Bet and Evaluation models retrained on new data: evaluating performance.")

        #NOTE: no longer removes data from replay buffer from version of bot which failed to
        #improve. I don't think this is significant, but see commits prior to 9/23/23 if interested.
        evaluateModelPerformance(curr_action_model, curr_bet_model, curr_eval_model,\
                                    best_action_model, best_bet_model, best_eval_model,\
                                    baseline_action_model, baseline_bet_model, baseline_eval_model,\
                                    iterations_without_improving, args)

        print(f"Evaluation complete: generating {num_new_transitions_before_eval_bet_training} new state transitions.")

if __name__ == "__main__":
    trainDQNAgent()