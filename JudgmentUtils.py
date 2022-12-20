def calcSubroundAdjustedValue(card,trump,secondary_suit):
    """
    Given a card, trump, and secondary suit, returns a simplified value for the card.

    Simplified values are defined as follows:

    Cards in the trump suit recieve a value from 14(2) - 26(Ace). This corresponds to the fact that
    even the lowest trump card beats the highest card from any other suit.

    Cards in the secondary suit recieve values from 1(2) - 13(Ace), corresponding to the fact they beat
    other suits but lose to even the lowest trump card.

    Finally, cards from all other suits recieve a value of 0, indicating that they cannot win the current subround.
    """
    if card.suit == trump:
        return (card.value-1)+13
    elif card.suit == secondary_suit:
        return (card.value-1)
    elif secondary_suit == None: #if there is no sceondary trump yet, this card is now secondary trump
        return (card.value-1)
    else:
        return 0