import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from search import get_search_query_embeddings

EMBEDDINGS_PATH = os.environ.get('EMBEDDING_TEMPLATE', "app/data/{}.npy")
ITEM_MASTER_PATH = os.environ.get('ENTITY_PATH_TEMPLATE', "app/data/Item_Master_{}.csv")


def create_embeddings(company_id, batch_size=100):
    s = time.time()
    items_df = pd.read_csv(ITEM_MASTER_PATH.format(company_id))
    corpus = sorted(list(set(items_df['product'])))

    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size)):
        temp_embeddings = get_search_query_embeddings(corpus[i:i + batch_size])
        corpus_embeddings.append(temp_embeddings)

    corpus_embeddings = np.concatenate(corpus_embeddings, axis=0)
    print(f'Embeddings Shape = {corpus_embeddings.shape}')
    np.save(EMBEDDINGS_PATH.format(company_id), corpus_embeddings)
    print("Embeddings created for Items_Master in: ", time.time() - s)


if __name__ == "__main__":
    create_embeddings(company_id=1005, batch_size=200)
