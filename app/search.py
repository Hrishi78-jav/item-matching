import os
import time
import numpy as np
import pandas as pd
import warnings
import torch.nn as nn
from utils.lambda_utils import invoke_lambda
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

warnings.filterwarnings('ignore')

USE_OPENSEARCH = int(os.environ.get('USE_OPENSEARCH', '0'))
EMBEDDINGS_PATH = os.environ.get('EMBEDDING_TEMPLATE', "data/{}.npy")
ITEM_MASTER_PATH = os.environ.get('ENTITY_PATH_TEMPLATE', "data/Item_Master_{}.csv")
ENTITY_SEARCH_LAMBDA = os.environ.get('ENTITY_SEARCH_LAMBDA', 'genie-entity-search-ml')
RERANK_MODEL_PATH = os.environ.get('RERANK_MODEL_PATH', 'model/miniLM_L6_100k')


def load_embeddings(company_id):
    """
    Loads saved embeddings for an entity

    :param (str) entity: Name of the entity for which embeddings are to be loaded
    :param (str) company_id: Identifier for the current company

    :return: (numpy.ndarray) Contains embedding corresponding to a particular entity
    """
    embeddings = np.load(EMBEDDINGS_PATH.format(company_id))
    return embeddings


def load_df(company_id):
    """
    Loads saved entity data

    :param (str) entity: Name of the entity for which data is to be loaded
    :param (int) company_id: Identifier for the current company

    :return: (pd.DataFrame) Loaded entity data
    """
    entity_df = pd.read_csv(ITEM_MASTER_PATH.format(company_id))
    return entity_df


def semantic_search(query_embeddings, corpus_embeddings, top_k=10):
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    top_matches_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    top_matches_scores = np.array([similarities[i][top_matches_indices[i]] for i in range(len(query_embeddings))])
    return top_matches_indices, top_matches_scores


def get_embeddings_from_lambda_output(lambda_output, embedding_dimension=384):
    num_list = lambda_output.split()
    result, temp = [], []
    for num in num_list:
        if num == '[' or num == ']' or num == '' or num == '],[':
            continue
        elif '[' not in num and ']' not in num:
            temp.append(float(num))
        elif '],[' == num[:3] or '],[' == num[-3:]:
            num = num.strip('],[')
            temp.append(float(num))
        elif '],[' in num and ('],[' != num[:3] or '],[' != num[-3:]):
            num = num.split('],[')
            temp += num
        elif ']' in num:
            num = num.strip(']')
            temp.append(float(num))
        elif '[' in num:
            num = num.strip('[')
            temp.append(float(num))
        if len(temp) >= embedding_dimension:
            result.append(temp[:embedding_dimension])
            temp = temp[embedding_dimension:]
    if len(temp) == embedding_dimension:
        result.append(temp)
    result = np.array(result)
    return result


def get_search_query_embeddings(query):
    """
    Creates query embeddings using pretrained Entity Search Model

    :param (str) query: Query for which embeddings need to be obtained
    :return: (np.ndarray) Query embeddings
    """
    search_input = {'input_text': query if type(query) == list else [query]}
    start = time.time()
    search_model_name = ENTITY_SEARCH_LAMBDA
    lambda_output = invoke_lambda(search_model_name, search_input)['query_embedding']
    query_embedding = get_embeddings_from_lambda_output(lambda_output)
    # query_embedding = np.array([float(item) for item in query_embedding.split(',')]).astype(np.float32)
    print("Time taken: ", time.time() - start)
    return query_embedding


def get_entity_search_output(query, company_id):
    query_embeddings = get_search_query_embeddings(query)
    corpus_embeddings = load_embeddings(company_id)
    items_df = load_df(company_id)
    corpus = sorted(list(set(items_df['product'])))
    top_matches_indices, top_matches_scores = semantic_search(query_embeddings, corpus_embeddings, top_k=10)
    search_output = []
    for i, indexes in enumerate(top_matches_indices):
        temp_corpus = [(corpus[index], top_matches_scores[i][j]) for j, index in enumerate(indexes)]
        search_output.append(temp_corpus)
    return search_output


def get_reranked_output(cross_encoder_model, query, corpus):
    if type(corpus[0]) == tuple:
        corpus = [element[0] for element in corpus]
    data = [[query, doc] for doc in corpus]
    pred = cross_encoder_model.predict(data)
    pred = sorted([[score, word] for score, word in zip(pred, corpus)], key=lambda x: x[0], reverse=True)
    pred = [x[1] for x in pred]
    return [pred[0]]


def get_final_output(queries, company_id):
    search_output = get_entity_search_output(queries, company_id)
    cross_encoder_model = CrossEncoder(RERANK_MODEL_PATH, num_labels=1, default_activation_function=nn.Sigmoid())
    final_output = []
    for i, query in enumerate(queries):
        final_output += get_reranked_output(cross_encoder_model, query, search_output[i])
    return final_output


if __name__ == '__main__':
    queries = ['fructo', 'thloda 10mg', 'dolo']
    output = get_final_output(queries, company_id=1005)
    print(output)
