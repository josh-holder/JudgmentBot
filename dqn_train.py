from collections import deque
import pickle
import os

def trainDQNAgent():
    print("hi")
    folder_name = "DQNRuns"
    act_mem_path = os.path.join(os.getcwd(),folder_name,"game_memory.pkl")
	if os.path.exists(mem_path):
		print("Loading existing experience data.")
		with open(mem_path,'rb') as f:
			memory = pickle.load(f)
	else:
		print("Previous experience data not found: generating empty memory list.")
		memory = deque(maxlen=nn_config.MEMORY_SIZE)