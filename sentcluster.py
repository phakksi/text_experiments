from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.metrics import get_scorer, silhouette_score


class BertEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, raw_documents):
        # Perform arbitary transformation
        model = SentenceTransformer(self.model_name)
        return model.encode(raw_documents)

# Evaluate metrics


def eval_metrics(self, actual, pred):

    results = defaultdict(None)

    for met in self.eval_metrics_list:
        if met == 'silhouette_score':
            results['silhouette_score'] = silhouette_score(
                self.embedded_sentences, pred)
        else:
            results[met] = get_scorer(met)._score_func(actual, pred)

    return results
