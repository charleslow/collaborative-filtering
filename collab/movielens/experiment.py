import numpy as np
import pandas as pd
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split

from .benchmark_utils import (
    DEFAULT_ITEM_COL,
    DEFAULT_K,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
    prepare_training_cornac,
    prepare_training_lightgcn,
    prepare_training_ncf,
    prepare_training_sar,
    prepare_training_svd,
    ranking_metrics_python,
    recommend_k_cornac,
    recommend_k_lightgcn,
    recommend_k_ncf,
    recommend_k_sar,
    recommend_k_svd,
    train_bivae,
    train_bpr,
    train_lightgcn,
    train_ncf,
    train_sar,
    train_svd,
)
from .params import params

# Params

environments = {
    "sar": "python_cpu",
    "svd": "python_cpu",
    "ncf": "python_cpu",
    "bpr": "python_cpu",
    "bivae": "python_cpu",
    "lightgcn": "python_cpu",
}

metrics = {
    "sar": ["ranking"],
    "svd": ["rating", "ranking"],
    "ncf": ["ranking"],
    "bpr": ["ranking"],
    "bivae": ["ranking"],
    "lightgcn": ["ranking"],
}

prepare_training_data = {
    "sar": prepare_training_sar,
    "svd": prepare_training_svd,
    "ncf": prepare_training_ncf,
    "bpr": prepare_training_cornac,
    "bivae": prepare_training_cornac,
    "lightgcn": prepare_training_lightgcn,
}

trainer = {
    "svd": lambda params, data: train_svd(params, data),
    "sar": lambda params, data: train_sar(params, data),
    "ncf": lambda params, data: train_ncf(params, data),
    "bpr": lambda params, data: train_bpr(params, data),
    "bivae": lambda params, data: train_bivae(params, data),
    "lightgcn": lambda params, data: train_lightgcn(params, data),
}

ranking_predictor = {
    "sar": lambda model, test, train: recommend_k_sar(model, test, train),
    "svd": lambda model, test, train: recommend_k_svd(model, test, train),
    "ncf": lambda model, test, train: recommend_k_ncf(model, test, train),
    "bpr": lambda model, test, train: recommend_k_cornac(model, test, train),
    "bivae": lambda model, test, train: recommend_k_cornac(model, test, train),
    "lightgcn": lambda model, test, train: recommend_k_lightgcn(model, test, train),
}

ranking_evaluator = {
    "sar": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "svd": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "ncf": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bivae": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "lightgcn": lambda test, predictions, k: ranking_metrics_python(
        test, predictions, k
    ),
}


def generate_summary(
    data,
    algo,
    k,
    train_time,
    time_rating,
    rating_metrics,
    time_ranking,
    ranking_metrics,
):
    summary = {
        "Data": data,
        "Algo": algo,
        "K": k,
        "Train time (s)": train_time,
        "Predicting time (s)": time_rating,
        "Recommending time (s)": time_ranking,
    }
    if rating_metrics is None:
        rating_metrics = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Explained Variance": np.nan,
        }
    if ranking_metrics is None:
        ranking_metrics = {
            "MAP": np.nan,
            "nDCG@k": np.nan,
            "Precision@k": np.nan,
            "Recall@k": np.nan,
        }
    summary.update(rating_metrics)
    summary.update(ranking_metrics)
    return summary


def experiment(data_size: str, algorithms: list[str]):
    cols = [
        "Data",
        "Algo",
        "K",
        "Train time (s)",
        "Predicting time (s)",
        "RMSE",
        "MAE",
        "R2",
        "Explained Variance",
        "Recommending time (s)",
        "MAP",
        "nDCG@k",
        "Precision@k",
        "Recall@k",
    ]
    df_results = pd.DataFrame(columns=cols)

    df = movielens.load_pandas_df(
        size=data_size,
        header=[
            DEFAULT_USER_COL,
            DEFAULT_ITEM_COL,
            DEFAULT_RATING_COL,
            DEFAULT_TIMESTAMP_COL,
        ],
    )
    print("Size of Movielens {}: {}".format(data_size, df.shape))

    # Split the dataset
    df_train, df_test = python_stratified_split(
        df,
        ratio=0.75,
        min_rating=1,
        filter_by="item",
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
    )

    # Loop through the algos
    for algo in algorithms:
        print(f"\nComputing {algo} algorithm on Movielens {data_size}")

        # Data prep for training set
        train = prepare_training_data.get(algo, lambda x, y: (x, y))(df_train, df_test)

        # Get model parameters
        model_params = params[algo]

        # Train the model
        model, time_train = trainer[algo](model_params, train)
        print(f"Training time: {time_train}s")

        # Predict and evaluate
        train, test = df_train, df_test

        ratings = None
        time_rating = np.nan

        if "ranking" in metrics[algo]:
            # Predict for ranking
            top_k_scores, time_ranking = ranking_predictor[algo](model, test, train)
            print(f"Ranking prediction time: {time_ranking}s")

            # Evaluate for rating
            rankings = ranking_evaluator[algo](test, top_k_scores, DEFAULT_K)
        else:
            rankings = None
            time_ranking = np.nan

        # Record results
        summary = generate_summary(
            data_size,
            algo,
            DEFAULT_K,
            time_train,
            time_rating,
            ratings,
            time_ranking,
            rankings,
        )
        df_results.loc[df_results.shape[0] + 1] = summary
