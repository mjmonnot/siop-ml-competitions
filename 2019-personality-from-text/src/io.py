import pandas as pd


def load_data(path):
    """
    Load the full competition file.
    Assumes Train / Dev / Test are identified by a 'Dataset' column.
    """
    df = pd.read_csv(path)

    if "Dataset" not in df.columns:
        raise ValueError("Expected a 'Dataset' column identifying Train/Dev/Test")

    return df


def split_by_dataset(df):
    train = df[df["Dataset"] == "Train"].copy()
    dev = df[df["Dataset"] == "Dev"].copy()
    test = df[df["Dataset"] == "Test"].copy()
    return train, dev, test
