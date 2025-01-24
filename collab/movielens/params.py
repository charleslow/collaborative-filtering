from benchmark_utils import (
    DEFAULT_ITEM_COL,
    DEFAULT_K,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
    SEED,
)

sar_params = {
    "similarity_type": "jaccard",
    "time_decay_coefficient": 30,
    "time_now": None,
    "timedecay_formula": True,
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
}
svd_params = {
    "n_factors": 150,
    "n_epochs": 15,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": SEED,
    "verbose": False,
}
ncf_params = {
    "model_type": "NeuMF",
    "n_factors": 4,
    "layer_sizes": [16, 8, 4],
    "n_epochs": 15,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "verbose": 10,
}
bpr_params = {
    "k": 200,
    "max_iter": 200,
    "learning_rate": 0.01,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False,
}
bivae_params = {
    "k": 100,
    "encoder_structure": [200],
    "act_fn": "tanh",
    "likelihood": "pois",
    "n_epochs": 500,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "seed": SEED,
    "use_gpu": True,
    "verbose": False,
}
lightgcn_param = {
    "model_type": "lightgcn",
    "n_layers": 3,
    "batch_size": 1024,
    "embed_size": 64,
    "decay": 0.0001,
    "epochs": 20,
    "learning_rate": 0.005,
    "eval_epoch": 5,
    "top_k": DEFAULT_K,
    "metrics": ["recall", "ndcg", "precision", "map"],
    "save_model": False,
    "MODEL_DIR": ".",
}
params = {
    "sar": sar_params,
    "svd": svd_params,
    "ncf": ncf_params,
    "bpr": bpr_params,
    "bivae": bivae_params,
    "lightgcn": lightgcn_param,
}
