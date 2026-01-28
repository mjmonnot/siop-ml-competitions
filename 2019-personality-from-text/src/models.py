from sklearn.linear_model import Ridge


def make_model(random_state):
    """
    Ridge regression is stable, variance-preserving,
    and works well under Pearson-r scoring.
    """
    return Ridge(alpha=1.0, random_state=random_state)
