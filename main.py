from collab.movielens.experiment import experiment

data_sizes = ["100k", "1m"]  # Movielens data size: 100k, 1m, 10m, or 20m
algorithms = ["svd", "sar", "ncf", "bpr", "bivae", "lightgcn"]

if __name__ == "__main__":
    experiment(data_size="100k", algorithms=algorithms)
