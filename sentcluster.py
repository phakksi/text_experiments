from typing import Iterable
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from utils.logging import logger


class SentenceClusterer():
    def __init__(self,
                 transformer_name: str = 'all-MiniLM-L6-v2') -> None:  # noqa: E501

        self.__transformer_name = transformer_name
        self.embeddings = None
        self.__transformer = SentenceTransformer(self.__transformer_name)
        self.__labels = None

    def __load_transformer(self):
        self.__transformer = SentenceTransformer(self.__transformer_name)

    def __apply_embeddings(self, sentences: Iterable[str]):
        self.embedddings = self.__transformer.encode(sentences)

    def set_transformer(self, transformer_name):
        self.__transformer_name = transformer_name
        self.__load_transformer()
        self.__apply_embeddings()

    def fit(self, sentences: Iterable[str], n_clusters: int = 5):
        if self.embeddings is None:
            self.embedddings = self.__transformer.encode(sentences)
        logger.info(len(self.embedddings))
        # km = KMeans(n_clusters)
        # logger.info(f'Training KMeans (n={n_clusters})')

        #self.__labels = km.fit_predict(self.embeddings)

        return self.__labels
