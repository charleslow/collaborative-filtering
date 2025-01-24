import math

import pandas as pd

DEFAULT_SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def get_splits(df: pd.DataFrame, splits_dict: dict, timestamp_col: str = "timestamp"):
    """
    Split dataframe based on timestamp percentiles for train, val and test.
    """
    assert sum(splits_dict.values()) == 1.0
    df = df.sort_values(by=timestamp_col)
    n = len(df)
    start_idx = 0
    split_dfs = {}
    for split_name, split_fraction in DEFAULT_SPLITS.items():
        end_idx = min(start_idx + math.ceil(n * split_fraction), n)
        split_dfs[split_name] = df.iloc[start_idx:end_idx]
        start_idx = end_idx
    return split_dfs


def load_raw_data(
    data_type: str = "ml-100k",
    threshold: int = 4,
    splits_dict: dict | None = None,
    drop_negatives: bool = True,
) -> dict[str, list[tuple]]:
    """
    Return as a list of ratings tuples of the form:
        (user_id, item_id, rating)
    """

    if data_type == "ml-100k":
        ratings = pd.read_csv(
            "data/ml-100k/u.data",
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

    ratings["rating"] = (ratings["rating"] >= threshold).astype("float32")
    if drop_negatives:
        ratings = ratings.loc[ratings["rating"] > 0.0]

    splits_dict = splits_dict if splits_dict else DEFAULT_SPLITS
    data = get_splits(ratings, splits_dict=splits_dict, timestamp_col="timestamp")
    for split_name, df in data.items():
        data[split_name] = list(
            df[["user_id", "item_id", "rating"]].itertuples(index=False, name=None)
        )
        print(f"Loaded data for {split_name} of len {len(data[split_name]):,}")
    return data


if __name__ == "__main__":
    data = load_raw_data()
