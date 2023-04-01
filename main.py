import pandas as pd
from sentcluster import SentenceClusterer
from utils.logging import logger


logger.info('Loading code...')
# load data
df = pd.read_csv('./data/yelp_labelled.txt', header=None,
                 sep='\t', names=['sentence', 'label'])
logger.info('done.')

sent_clu = SentenceClusterer()
labels = sent_clu.fit(list(df.head(10).sentence))
# print(labels)
