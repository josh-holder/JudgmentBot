from multiprocessing import Process, cpu_count, Pool, Manager
from DQNAgent import DQNAgent
from HumanBetAgent import HumanBetAgent
from JudgmentGame import JudgmentGame
import time
import os

def gamePlayingProcess(i, num_games, data_dict):
    # jg = JudgmentGame(agents=[DQNAgent(0),DQNAgent(1),DQNAgent(2),DQNAgent(3)])
    jg = JudgmentGame(agents=[HumanBetAgent(0,use_eval_model=True),HumanBetAgent(1,use_eval_model=True),HumanBetAgent(2,use_eval_model=True),HumanBetAgent(3,use_eval_model=True)])

    data = []
    for game_num in range(num_games):
        print(f"Agent {i}, game {game_num}")
        data_item = jg.playGameAndTrackStateTransitions()
        data.append(data_item)

        jg.resetGame()
    
    data_dict[i] = data

def joinDataFromThreadedGames(threaded_data_dict):
    """
    Recieve a dict containing data from several processes playing JudgmentGames.
    Dictionary is of the form {process_1_name: data_1, process_2_name: data_2, ...}
    and data_i is of the form:

    data_i = [(bet_data_agenti_game1, eval_data_agenti_game0, act_data_agenti_game0),
                (bet_data_agenti_game1, eval_data_agenti_game1, act_data_agenti_game1),
                ...
                (bet_data_agenti_gamen, eval_data_agenti_gamen, act_data_agenti_gamen)]

    Want to combine all this data into large lists of the form

    bet_data_list = [bet_data_1_0, bet_data_1_1, ..., bet_data_1_n, bet_data_]
    """
    master_bet_data_list = []
    master_eval_data_list = []
    master_act_data_list = []
    for agent, agent_data in threaded_data_dict.items():
        for data_from_game in agent_data:
            bet_data_from_game = data_from_game[0]
            master_bet_data_list.extend(bet_data_from_game)

            eval_data_from_game = data_from_game[1]
            master_eval_data_list.extend(eval_data_from_game)

            act_data_from_game = data_from_game[2]
            master_act_data_list.extend(act_data_from_game)

    return master_bet_data_list, master_eval_data_list, master_act_data_list

if __name__ == "__main__":
    processes = []
    start = time.time()

    # jg = JudgmentGame(agents=[DQNAgent(1),DQNAgent(2),DQNAgent(3),DQNAgent(3)])
    # for i in range(9):
    #     jg.playGameAndCollectSLData()
    #     jg.resetGame()
    
    # print(time.time()-start)

    game_results_manager = Manager()
    results_dict = game_results_manager.dict()

    for i in range(cpu_count()-1):
        p = Process(target=gamePlayingProcess, args=(i, 2, results_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print(time.time()-start)

    print("Top level:")
    print(type(results_dict[0]))
    print(f"num keys {len(results_dict.keys())}")

    print("data from agent 0")
    print(type(results_dict[0]))
    print(f"num games {len(results_dict[0])}")
    print(f"len of data in agent 0 {len(results_dict[0][0])}")
    print(f"len of data in agent 0 {len(results_dict[0][1])}")

    print("len of bet data")
    print(len(results_dict[0][0]))
    print(len(results_dict[0][1]))
    # print(results_dict.keys())
    # print(results_dict[0][0][0])

    master_bet, master_eval, master_act = joinDataFromThreadedGames(results_dict)
    print(len(master_bet),len(master_eval),len(master_act))
    print(master_bet[0])
    print("-------")
    print(master_eval[0])
    print("-------")
    print(master_act[60])