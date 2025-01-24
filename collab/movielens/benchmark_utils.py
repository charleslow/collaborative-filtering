# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
Taken from:
https://github.com/recommenders-team/recommenders/blob/main/examples/06_benchmarks/benchmark_utils.py
"""

import os
from tempfile import TemporaryDirectory

import cornac
import pandas as pd
import surprise
from recommenders.evaluation.python_evaluation import (
    exp_var,
    mae,
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    rsquared,
)
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.models.sar import SAR
from recommenders.models.surprise.surprise_utils import (
    compute_ranking_predictions,
    predict,
)
from recommenders.utils.constants import (
    COL_DICT,
    DEFAULT_ITEM_COL,
    DEFAULT_K,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
    SEED,
)
from recommenders.utils.timer import Timer

try:
    from recommenders.evaluation.spark_evaluation import (
        SparkRankingEvaluation,
        SparkRatingEvaluation,
    )
    from recommenders.utils.spark_utils import start_or_get_spark
except (ImportError, NameError):
    pass  # skip this import if we are not in a Spark environment
try:
    from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
    from recommenders.models.deeprec.deeprec_utils import prepare_hparams
    from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
    from recommenders.models.ncf.dataset import Dataset as NCFDataset
    from recommenders.models.ncf.ncf_singlenode import NCF
except ImportError:
    pass  # skip this import if we are not in a GPU environment

# Helpers
tmp_dir = TemporaryDirectory()
TRAIN_FILE = os.path.join(tmp_dir.name, "df_train.csv")
TEST_FILE = os.path.join(tmp_dir.name, "df_test.csv")


def prepare_training_svd(train, test):
    reader = surprise.Reader("ml-100k", rating_scale=(1, 5))
    return surprise.Dataset.load_from_df(
        train.drop(DEFAULT_TIMESTAMP_COL, axis=1), reader=reader
    ).build_full_trainset()


def train_svd(params, data):
    model = surprise.SVD(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def predict_svd(model, test):
    with Timer() as t:
        preds = predict(
            model,
            test,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
        )
    return preds, t


def recommend_k_svd(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = compute_ranking_predictions(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t


def prepare_training_ncf(df_train, df_test):
    train = df_train.sort_values(["userID"], axis=0, ascending=[True])
    test = df_test.sort_values(["userID"], axis=0, ascending=[True])
    test = test[df_test["userID"].isin(train["userID"].unique())]
    test = test[test["itemID"].isin(train["itemID"].unique())]
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    return NCFDataset(
        train_file=TRAIN_FILE,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        seed=SEED,
    )


def train_ncf(params, data):
    model = NCF(n_users=data.n_users, n_items=data.n_items, **params)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_ncf(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        users, items, preds = [], [], []
        item = list(train[DEFAULT_ITEM_COL].unique())
        for user in train[DEFAULT_USER_COL].unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))
        topk_scores = pd.DataFrame(
            data={
                DEFAULT_USER_COL: users,
                DEFAULT_ITEM_COL: items,
                DEFAULT_PREDICTION_COL: preds,
            }
        )
        merged = pd.merge(
            train, topk_scores, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="outer"
        )
        topk_scores = merged[merged[DEFAULT_RATING_COL].isnull()].drop(
            DEFAULT_RATING_COL, axis=1
        )
    # Remove temp files
    return topk_scores, t


def prepare_training_cornac(train, test):
    return cornac.data.Dataset.from_uir(
        train.drop(DEFAULT_TIMESTAMP_COL, axis=1).itertuples(index=False), seed=SEED
    )


def recommend_k_cornac(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = predict_ranking(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t


def train_bpr(params, data):
    model = cornac.models.BPR(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def train_bivae(params, data):
    model = cornac.models.BiVAECF(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def prepare_training_sar(train, test):
    return train


def train_sar(params, data):
    model = SAR(**params)
    model.set_index(data)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_sar(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


def prepare_training_lightgcn(train, test):
    return ImplicitCF(train=train, test=test)


def train_lightgcn(params, data):
    hparams = prepare_hparams(**params)
    model = LightGCN(hparams, data)
    with Timer() as t:
        model.fit()
    return model, t


def recommend_k_lightgcn(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


def rating_metrics_pyspark(test, predictions):
    rating_eval = SparkRatingEvaluation(test, predictions, **COL_DICT)
    return {
        "RMSE": rating_eval.rmse(),
        "MAE": rating_eval.mae(),
        "R2": rating_eval.exp_var(),
        "Explained Variance": rating_eval.rsquared(),
    }


def ranking_metrics_pyspark(test, predictions, k=DEFAULT_K):
    rank_eval = SparkRankingEvaluation(
        test, predictions, k=k, relevancy_method="top_k", **COL_DICT
    )
    return {
        "MAP": rank_eval.map(),
        "nDCG@k": rank_eval.ndcg_at_k(),
        "Precision@k": rank_eval.precision_at_k(),
        "Recall@k": rank_eval.recall_at_k(),
    }


def rating_metrics_python(test, predictions):
    return {
        "RMSE": rmse(test, predictions, **COL_DICT),
        "MAE": mae(test, predictions, **COL_DICT),
        "R2": rsquared(test, predictions, **COL_DICT),
        "Explained Variance": exp_var(test, predictions, **COL_DICT),
    }


def ranking_metrics_python(test, predictions, k=DEFAULT_K):
    return {
        "MAP": map(test, predictions, k=k, **COL_DICT),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **COL_DICT),
        "Precision@k": precision_at_k(test, predictions, k=k, **COL_DICT),
        "Recall@k": recall_at_k(test, predictions, k=k, **COL_DICT),
    }


def ranking_metrics_python(test, predictions, k=DEFAULT_K):
    return {
        "MAP": map(test, predictions, k=k, **COL_DICT),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **COL_DICT),
        "Precision@k": precision_at_k(test, predictions, k=k, **COL_DICT),
        "Recall@k": recall_at_k(test, predictions, k=k, **COL_DICT),
    }
