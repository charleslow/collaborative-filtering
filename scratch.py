import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Step 1: Load and preprocess the dataset
ratings = pd.read_csv(
    "data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
)
user_item_matrix = csr_matrix(
    (ratings["rating"], (ratings["user_id"], ratings["item_id"]))
)
user_item_matrix = (user_item_matrix > 0).astype(np.float32)
from pdb import set_trace


set_trace()
