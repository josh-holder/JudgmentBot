# JudgmentBot
Tools to create a machine learning AI capable of playing the card game Judgment.

## Play the bot!

If you want to try your hand at beating the bot, install required packages with `pip install -r requirements.txt`, and run the following command:

`python3 play_game_against_bots.py`

following the instructions in the command line. Best of luck - it might surprise you!

## Terminology

There is a lot of somewhat subtle terminology used throughout the code base.

### Game vs. Round vs. Subround
In a trick-taking card game, there are often overarching rounds in which each player has multiple cards in their hand, and plays them until they eventually run out of cards. This is a "round."

However, in the process of a round, there is often a structure in which each agent plays a single card, and the player which played the highest of these cards wins the cards involved or some other reward. This is termed a "subround".

Finally, a "game" consists of many rounds (i.e. Judgment consists of rounds where players have anywhere from 1 to 13 to 1 cards.)

### Action agent
This is the neural network which evaluates the expected point value of playing a card in a given situation.

The input to this model is an Action state (see below), and the output is an expected value. NNAgent evaluates this function on all cards in its hand, and selects the one with the highest value (or selects epsilon-greedily.)

### Evaluation agent
This is a neural network which performs an intermediate task - it evaluates the odds which playing a card will win you the round.

This information is used in the action network to determine how smart of an idea playing a given card is (i.e. if the network thinks there is a good chance that a card will win the round and you're betting zero, it will not evaluate this option as a good one.)

The advantage of having this network be an explicit component of another network is that we can get a clear and fast-acting learning signal from this - after each subround, we get a clear answer on whether or not the card led to a subround win or not.

### SubroundSituation
Aims to be a self-contained object which contains all the information needed to fully define a subround sitation. Using this info combined with info about its own hand, any agent can determine what card to play in a given situation.

Includes info on hand size, the cards on the stack, the trump suit, the other agents in the game, and publicly played cards.

### Action state
All the information about an action such that it can be fed into a neural network and evaluated.

This contains information about the SubroundSituation, as well as about the agent itself and the card it has chosen to play.

### Evaluation state
All the information about an action such that it can be fed into a neural network and evaluated.

The only difference between an eval state and an action state is that the only informatioin relevant to "will this card win the subround" is the value of the card - other considerations (i.e. should I not be throwing out the ace of spades if I'm trying to win rounds in the future) are not relevant to this. This allows us to cut down the inputs by ~50%.

### BetSituation
Self-contained object which contains all information about the state of the game when betting.

### BetState
You know the drill at this point.

## JudgmentGame.py

Defines the class JudgmentGame which simulates Judgment games with arbitrary agents.

You'll notice there are two methods for playing games - one tracks data for use in training ML models, and another simply plays games without tracking action states. For fast evaluation, use playGame().

## JudgmentAgent.py

Defines the generic class JudgementAgent which can participate in a game of Judgement.

Any classes inheriting from the JudgmentAgent class must define their own makeBet and chooseCard methods which are more than purely random.

## judgment_value_models.py

File which determines structure of betting, action, and evaluation models.

## dqn_train.py

File which defines the training workflow for DQN agents.