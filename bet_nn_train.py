import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import argparse
from bet_data_generator import loadBetData, betSituation

def _build_parser():
    parser = argparse.ArgumentParser(description='Train ML agent to make judgement bets.')

    return parser

def prepareData():
    #load bet data and convert to appropriate form 
    # [spades - hearts - diamonds - clubs - # ppl already bet - # zero bets - percent of total - trump]
    raw_bet_data = loadBetData()
    prepared_data_in = -0.5*np.ones((len(raw_bet_data),56))
    prepared_data_out = np.zeros((len(raw_bet_data),1))

    for i,bet in enumerate(raw_bet_data):
        bs = bet[0]

        #add cards
        for card in bs.cards:
            data_index = 13*card.suit + card.value - 2
            prepared_data_in[i,data_index] = 0.5

        prepared_data_in[i,52] = bs.bet_pos
        prepared_data_in[i,53] = bs.other_bets.count(0)
        prepared_data_in[i,54] = sum(bs.other_bets)/bs.round
        prepared_data_in[i,55] = bs.trump
    
        #have it output a percentage of hands in the round to bet
        prepared_data_out[i] = bet[1]/bs.round

    return [prepared_data_in, prepared_data_out]

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    prepared_data = prepareData()
    prepared_data_in = prepared_data[0]
    prepared_data_out = prepared_data[1]

    model_path = os.path.join(os.getcwd(),'bet_agent')
    if os.path.exists(model_path):
        print("Loading existing NN.")
        model = keras.models.load_model(model_path)
    else:
        print("Previous NN not found: generating NN from scratch.")
        model = tf.keras.models.Sequential([
            keras.Input(shape=(56)),
            layers.Dense(32,activation='relu'),
            layers.Dense(16,activation='relu'),
            layers.Dense(16,activation='relu'),
            layers.Dense(1),
        ])
        model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())
        model.summary()

    data_split = 0.7
    split_idx = int(data_split*np.size(prepared_data_in,0))
    
    train_in = prepared_data_in[:split_idx,:]; test_in = prepared_data_in[split_idx:,:];
    train_out = prepared_data_out[:split_idx,:].flatten(); test_out = prepared_data_out[split_idx:,:].flatten();

    # model.fit(train_in,train_out,epochs=50,batch_size=50)
    # model.evaluate(test_in, test_out, verbose=2)

    # model.save(model_path)
    print(model.predict(test_in))


