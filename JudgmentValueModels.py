import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Masking
from tensorflow.keras import Input, Model

def initBetModel(layer_sizes=[48,48,24]):
    """
    Model which, given bet situation, will output the expected average value of the bet relative to the
    other players (normalized by 130, which is the maximum point difference).
    """
    param_input = Input(60)

    input_layer = param_input
    for i,layer in enumerate(layer_sizes):
        MLP_layer = layers.Dense(layer,activation="relu")(input_layer)
        if i % 2 == 0:
            adjust_layer = layers.Dropout(0.5)(MLP_layer)
        else:
            adjust_layer = layers.BatchNormalization()(MLP_layer)

        input_layer = adjust_layer

    bet_value = layers.Dense(1,activation="tanh")(input_layer)

    model = Model(inputs=param_input,outputs=bet_value)
    model.compile(loss = tf.losses.MeanSquaredError(),
                    optimizer = tf.optimizers.Adam())
    return model

def initEvalModel(layer_sizes=[48,24,12]):
    """
    Model which, given playing situation, outputs the probability that playing a given card will result
    in winning the hand.

    Inputs:
    -LSTM output of the players going after you in the order. Each timestep in the LSTM contains the following info:
        - Player relative points
        - Player bet
        - Player earned (normalized)
        - Suits remaining (4 binary values)
    - Cards still possible to play (52 binary values, either -1 or 1)
    - Trump suit (4 binary values, one-hot encoding)
    - Value of card to play, representing action (0 if off-suit, 1-13 if secondary suit, 14-26 if trump suit)
    - Agent bet (normalized to 13)
    - Agent subrounds already won (normalized to 13)
    - Number of cards in hand
    """
    #Input for LSTM of players going after you in the order
    next_players_input = Input(shape=(3,7))
    next_players_mask = Masking(mask_value=-1.0)(next_players_input)
    next_players_LSTM = layers.LSTM(20, activation="relu", unroll=True)(next_players_mask)

    #Input for all other parameters, including cards still possible to play, trump, player info, etc.
    parameters = Input(shape=60)

    #Concatenate LSTM and regular parameters into fully connected layers
    all_params = concatenate([next_players_LSTM, parameters])

    #Add the specified amount and size of fully connected layers
    input_layer = all_params
    for i,layer in enumerate(layer_sizes):
        MLP_layer = layers.Dense(layer,activation="relu")(input_layer)
        if i % 2 == 0:
            adjust_layer = layers.Dropout(0.5)(MLP_layer)
        else:
            adjust_layer = layers.BatchNormalization()(MLP_layer)

        input_layer = adjust_layer

    #Predict percentage win value
    win_value = layers.Dense(1,activation="sigmoid")(input_layer)

    model = Model(inputs=[next_players_input,parameters],outputs=win_value)
    model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

    return model

def initActionModel(layer_sizes=[128,64,32,16]):
    """
    Model which, given a playing situation, outputs the expected values of all the cards.

    Inputs:
    -LSTM output of the players going after you in the order. Each timestep in the LSTM contains the following info:
        - Player relative points
        - Player bet
        - Player earned (normalized)
        - Suits remaining (4 binary values)
    -Information on the current winner of the round
        - Player relative points
        - Player bet
        - Player earned (normalized)
    - Cards still possible to play (52 binary values, either -1 or 1)
    - Card that you're about to play (one-hot encoding with 52 binary values)
    - Trump suit (4 binary values, also one-hot encoding)
    - Agent bet (normalized to 13)
    - Agent subrounds already won (normalized to 13)
    - Chance that that card is going to win
    - Number of cards in hand

    Output:
    Expected value of the action relative to other players, normalized to 260.
    """
    #Input for LSTM of players going after you in the order
    next_players_input = Input(shape=(3,7))
    next_players_mask = Masking(mask_value=-1.0)(next_players_input)
    next_players_LSTM = layers.LSTM(20, activation="relu", unroll=True)(next_players_mask)

    #Input for currently winning player (need masking here if there is no winning player)
    winning_player_input = Input(shape=(3))
    winning_player_mask = Masking(mask_value=-1.0)(winning_player_input) #masking value can't be 0 because all zeros is plausible here
    
    #Input for other parameters (cards not played, action, trump, agent info, etc.)
    parameters = Input(shape=112)

    #Concatenate LSTM, winning player, and regular parameters into fully connected layers
    all_params = concatenate([next_players_LSTM, winning_player_mask, parameters])

    #Add the specified amount and size of fully connected layers
    input_layer = all_params
    for i,layer in enumerate(layer_sizes):
        MLP_layer = layers.Dense(layer,activation="relu")(input_layer)
        if i % 2 == 0:
            adjust_layer = layers.Dropout(0.5)(MLP_layer)
        else:
            adjust_layer = layers.BatchNormalization()(MLP_layer)

        input_layer = adjust_layer

    #Predict action value
    action_value = layers.Dense(1,activation="tanh")(input_layer)

    model = Model(inputs=[next_players_input,winning_player_input,parameters],outputs=action_value)
    model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

    return model

if __name__ == "__main__":
    eval_model = initEvalModel()
    print(eval_model.summary())

    bet_model = initBetModel()
    print(bet_model.summary())

    action_model = initActionModel()
    print(action_model.summary())