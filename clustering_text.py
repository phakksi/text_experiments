# %%
from sys import version_info
from sentcluster import BertEmbedder
from utils.logging import logger
import pandas as pd
import numpy as np
import mlflow
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import cloudpickle
import warnings
import time

# %%
# RANDOM_SEED
RANDOM_SEED = 42

# CLUSTER
CLUSTER_MIN = 5
CLUSTER_MAX = 15
CLUSTER_STEP = 1

LOG_ONLY_BEST = True

# data
DATA_PATH = './data/yelp_labelled.txt'

# sentence transformer
# choose one from https://www.sbert.net/docs/pretrained_models.html
MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1'

# Eval Metrics
EVAL_METRICS = ['silhouette_score']

N_REVIEW_SAMPLES = 5
# %%
# load data
logger.info('Loading file {DATA_PATH}...')
df = pd.read_csv(DATA_PATH, header=None,
                 sep='\t', names=['sentence', 'label'])
logger.info(f'Load complete: {df.shape[0]} rows and {df.shape[1]} columns.')

# %%
logger.info(f'Loading sentence transformer {MODEL_NAME}...')
embedder = BertEmbedder(MODEL_NAME)
logger.info('done.')


# %% [markdown]
# # Config conda env

# %%
PYTHON_VERSION = "{major}.{minor}.{micro}".format(

    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)

conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": ["python={}".format(PYTHON_VERSION), "pip"],
    "pip": [
        "mlflow",
        "cloudpickle=={}".format(cloudpickle.__version__),
        "scikit-learn==1.2.2",
        "sentence-transformers==2.2.2"
    ],
    "name": "mlflow-env",
}


# %% [markdown]
# # Training Step

# %%


np.random.seed(RANDOM_SEED)

pipe = Pipeline([('embedder', embedder), ('clusterer', KMeans())])

best_metric = np.Inf
best_labels = None

for current_n in range(CLUSTER_MIN, CLUSTER_MAX, CLUSTER_STEP):
    with mlflow.start_run():
        warnings.filterwarnings("ignore")
        mlflow.log_param("n_clusters", current_n)
        model_start_time = time.strftime("%Y%m%d-%H%M%S")
        pipe.set_params(**{'clusterer__n_clusters': current_n})
        predicted_2d = pipe.fit(df.sentence)
        predicted_labels = pipe.predict(df.sentence)

        current_metric = silhouette_score(
            pipe['embedder'].transform(df.sentence), predicted_labels)

        logger.info(
            f"n_cluster={current_n}, silhouette_score={current_metric:.5f}")

        if LOG_ONLY_BEST and current_metric < best_metric:
            best_metric = current_metric
            mlflow.sklearn.log_model(
                pipe, f"{MODEL_NAME}_KMeans_(n={current_n})", conda_env=conda_env)
            mlflow.log_metric('silhouette_score', best_metric)
            best_labels = predicted_labels
        else:
            mlflow.sklearn.log_model(
                pipe, f"{MODEL_NAME}_KMeans_(n={current_n})", conda_env=conda_env)
            mlflow.log_metric('silhouette_score', current_metric)
            best_labels = predicted_labels


# %% [markdown]
# # Generate scored dataset with best model

# %%
results_df = pd.DataFrame({'sentence': df.sentence, 'label': best_labels})

# %%
for r in sorted(results_df.label.unique()):
    logger.info('------------------------------------')
    logger.info(f'samples from cluster={r}')
    logger.info('------------------------------------')
    for index, row in results_df[results_df.label == r].head(N_REVIEW_SAMPLES).iterrows():
        logger.info(row['sentence'])

# %%
