def calcSubroundAdjustedValue(card,srs):
    """
    Given a card and subround situation, returns a simplified value for the card.

    Simplified values are defined as follows:

    Cards in the trump suit recieve a value from 14(2) - 26(Ace). This corresponds to the fact that
    even the lowest trump card beats the highest card from any other suit.

    Cards in the secondary suit recieve values from 1(2) - 13(Ace), corresponding to the fact they beat
    other suits but lose to even the lowest trump card.

    Finally, cards from all other suits recieve a value of 0, indicating that they cannot win the current subround.
    """
    trump_suit = srs.trump
    secondary_suit = srs.card_stack[0].suit if (len(srs.card_stack) > 0) else None

    if card.suit == trump_suit:
        return (card.value)+13
    elif card.suit == secondary_suit:
        return (card.value)
    elif secondary_suit == None: #if there is no sceondary trump yet, this card is now secondary trump
        return (card.value)
    else:
        return 0