import pandas as pd


def load_raw_data(data_type: str = "ml-100k", threshold: int = 4) -> list[tuple]:
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
    tuples = list(
        ratings[["user_id", "item_id", "rating"]].itertuples(index=False, name=None)
    )
    return tuples


if __name__ == "__main__":
    data = load_raw_data()
