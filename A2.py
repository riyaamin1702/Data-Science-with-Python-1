#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from numpy.linalg import norm
import dask.dataframe as dd
from dask.multiprocessing import get
import matplotlib.pyplot as plt
import re
import tqdm
import pickle
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.keyedvectors import KeyedVectors
import time
from sklearn.metrics import ndcg_score
from sklearn.metrics import make_scorer

from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# In[2]:


CORES = 12


# # Feature extraction in documents and queries (one time task)


# break documents.tsv to chunks for easy loading
def break_document_into_chunks(chunk_size=1000):
    import os
    try:
        os.mkdir('docs_chunked')
    except:
        pass
    batch_no = 1
    df_chunk_map = pd.DataFrame(columns=['Docid', 'chunk'])
    for chunk in pd.read_table('documents.tsv', chunksize=chunk_size, header=None):
        chunk_name = 'docs_chunked/chunk' + str(batch_no) + '.tsv'
        print(chunk_name)
        chunk.columns = ['Docid', 'html_data', 'parse_text']
        chunk.to_csv(chunk_name, sep=',')
        batch_no += 1
        df_chunk_map = pd.concat(
            [df_chunk_map, pd.DataFrame({'Docid': chunk.iloc[:, 0], 'chunk': [chunk_name] * len(chunk)})])
    df_chunk_map = df_chunk_map.set_index("Docid")
    df_chunk_map.to_csv('docs_chunked/docid2chunk.csv')


# function to load document chunk

def load_doc_chunk(docid2chunk, docids=None, chunkno=-1, preprocessed=False):
    def read_chunk(fname):
        df = pd.read_csv(fname)
        df.columns = ['Docid', 'processed_data'] if preprocessed else ['drp', 'Docid', 'html_data', 'parse_text']
        df = df[['Docid', 'processed_data']] if preprocessed else df[['Docid', 'html_data', 'parse_text']]
        return df.set_index('Docid')

    if chunkno > 0:
        return read_chunk(f"docs_chunked/{'p' if preprocessed else ''}chunk{chunkno}.tsv")
    if docids is not None:
        chunk_names = docid2chunk.loc[docids, 'chunk'].unique()
        if preprocessed:
            chunk_names = [re.sub('chunk', 'pchunk', a) for a in chunk_names]
        df = read_chunk(chunk_names[0])
        for i in range(1, len(chunk_names)):
            df = pd.concat(df, read_chunk(chunk_names[i]))
        return df
    raise Exception("ERR")


# Dictionary of english Contractions
contractions_dict = {"ain't": "are not", "'s": " is", "aren't": "are not", "can't": "can not",
                     "can't've": "cannot have",
                     "'cause": "because", "could've": "could have", "couldn't": "could not",
                     "couldn't've": "could not have",
                     "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                     "hadn't've": "had not have",
                     "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
                     "he'll": "he will",
                     "he'll've": "he will have", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                     "i'd": "i would",
                     "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                     "i've": "i have",
                     "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                     "it'll've": "it will have",
                     "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                     "mightn't": "might not",
                     "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                     "mustn't've": "must not have",
                     "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                     "oughtn't": "ought not",
                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                     "she'll": "she will",
                     "she'll've": "she will have", "should've": "should have", "shouldn't": "should not",
                     "shouldn't've": "should not have", "so've": "so have", "that'd": "that would",
                     "that'd've": "that would have",
                     "there'd": "there would", "there'd've": "there would have",
                     "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                     "they'll've": "they will have",
                     "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                     "we'd": "we would",
                     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                     "we've": "we have",
                     "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                     "what're": "what are",
                     "what've": "what have", "when've": "when have", "where'd": "where did",
                     "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                     "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                     "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                     "y'all": "you all",
                     "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                     "y'all've": "you all have",
                     "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                     "you'll've": "you will have",
                     "you're": "you are", "you've": "you have"}


# Function for expanding contractions
def expand_contractions(text):
    def replace(match):
        return contractions_dict[match.group(0)]

    # Regular expression for finding contractions
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    return contractions_re.sub(replace, text)


# Function for Cleaning Text
def clean_text(text):
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub('[^a-z]', ' ', text)
    return text


# # Stopwords removal & Lemmatizing tokens using SpaCy
# import spacy
# nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])
# nlp.max_length=5000000


def preprocess_doc(chunk):
    chunk['processed_data'] = chunk['parse_text'].apply(lambda x: expand_contractions(x.lower()))
    chunk['processed_data'] = chunk['processed_data'].apply(lambda x: clean_text(x))
    chunk['processed_data'] = chunk['processed_data'].apply(lambda x: re.sub(' +', ' ', x))  # Removing extra spaces
    # Removing Stopwords and Lemmatizing words
    #     chunk['lemmatized'] = chunk['processed_data'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))
    return chunk


def preprocess_query(q):
    q['processed_query'] = q['query'].apply(lambda x: x.lower())  # Lowercasing the text
    q['processed_query'] = q['processed_query'].apply(lambda x: expand_contractions(x))  # Expanding contractions
    q['processed_query'] = q['processed_query'].apply(lambda x: clean_text(x))  # Cleaning queries using RegEx
    q['processed_query'] = q['processed_query'].apply(lambda x: re.sub(' +', ' ', x))  # Removing extra spaces
    return q


def preprocess_all_chunks(docid2chunk):
    for nc in tqdm.tqdm(range(len(docid2chunk['chunk'].unique()))):
        chunk = load_doc_chunk(docid2chunk, chunkno=nc + 1)
        chunk_pdf = preprocess_doc(chunk)
        chunk_pdf = chunk_pdf[['processed_data']]
        chunk_name = 'docs_chunked/pchunk' + str(nc + 1) + '.tsv'
        chunk_pdf.to_csv(chunk_name, sep=',')


# train Word2Vec model using training data


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.epoch_start_time = time.time()

    def on_epoch_end(self, model):
        print("Epoch #{} end dt={}".format(self.epoch, time.time() - self.epoch_start_time))
        self.epoch_start_time = time.time()
        model.wv.save_word2vec_format('w2v_model.bin', binary=True)
        self.epoch += 1


def create_w2v_model():
    from gensim.models import Word2Vec, KeyedVectors

    queries_df = pd.read_table('queries.tsv', header=None)
    queries_df.columns = ['qid', 'query']

    train_docids = pd.read_table('train.tsv')['Docid'].unique()
    train_queryids = pd.read_table('train.tsv')['#QueryID'].unique()

    pqueries_df = preprocess_query(queries_df)
    pqueries_df = pqueries_df[pqueries_df['qid'].isin(train_queryids)].reset_index(drop=True)

    combined_training = pqueries_df.rename(columns={'processed_query': 'text'})['text']

    for nc in tqdm.tqdm(range(len(docid2chunk['chunk'].unique()))):
        pchunk = load_doc_chunk(docid2chunk, chunkno=nc + 1, preprocessed=True)
        # select only docs in training set
        pchunk = pchunk[pchunk.index.isin(train_docids)]

        # Combining training data and queries for training
        combined_training = pd.concat([pchunk.rename(columns={'processed_data': 'text'})['text'], combined_training])

        if nc > 10:  # break here because can't load all. Too much time to train!!
            break

    combined_training = combined_training.sample(frac=1)

    # Creating data for the model training
    train_data = []
    for i in combined_training:
        train_data.append(i.split())
    print('Total words', len(train_data))

    # Training a word2vec model from the given data set
    epoch_logger = EpochLogger()
    w2v_model = Word2Vec(train_data, vector_size=300, min_count=2, window=5, sg=1, workers=CORES,
                         callbacks=[epoch_logger])

    w2v_model.wv.save_word2vec_format('w2v_model.bin', binary=True)


# Function returning vector representation of a document
def get_embedding_w2v(w2v_model, doc_tokens):
    embeddings = []
    if len(doc_tokens) < 1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.key_to_index:
                embeddings.append(w2v_model.get_vector(tok))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)


# Saving Word2Vec Vectors for all documents and queries
def convert_queries_and_doc_chunks_to_vectors(docid2chunk, w2v_model):
    # converting doc chunks to vectors
    doc_vec = pd.Series([], )
    doc_vec.index.name = 'Docid'
    for nc in tqdm.tqdm(range(len(docid2chunk['chunk'].unique()))):
        pchunk = load_doc_chunk(docid2chunk, chunkno=nc + 1, preprocessed=True)
        #         ddata = dd.from_pandas(pchunk, npartitions=CORES)
        #         res = ddata.map_partitions(lambda df: df.apply((lambda row: get_embedding_w2v((row['processed_data'].split()))), axis=1)).compute(scheduler='threads')

        res = pchunk['processed_data'].apply(lambda x: get_embedding_w2v(w2v_model, x.split()))

        doc_vec = doc_vec.append(res)
    doc_vec = pd.DataFrame(doc_vec).rename(columns={0: 'vector'})
    doc_vec.to_csv('documents_vec.csv')

    # converting queries to vectors
    queries_df = pd.read_table('queries.tsv', header=None)
    queries_df.columns = ['qid', 'query']
    pqueries_df = preprocess_query(queries_df)
    pqueries_df['vector'] = pqueries_df['processed_query'].apply(lambda x: get_embedding_w2v(w2v_model, x.split()))
    pqueries_df = pqueries_df[['qid', 'vector']].set_index('qid')
    pqueries_df.to_csv('queries_vec.csv', sep=',')


def cosine_similarity(a, b):
    return np.dot(a, b.T) / (norm(a) * norm(b))


def NDCG_loss_func(y, y_pred):
    true_relevance = [np.array(y)]
    scores = [np.array(y_pred)]
    #     print(true_relevance, scores, len(y),len(y_pred))
    return ndcg_score(true_relevance, scores)


def get_model_and_paramgrid():
    model = LinearRegression()
    params_grid = {
        'fit_intercept': [True, False],
        'positive': [False]
    }

    # model = LinearSVR()
    # params_grid = {
    #         'C': [0.1, 1.0, 5., 10.],
    #         'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    # }
    print(model.get_params().keys())

    return model, params_grid


def ranking_ir(model, query, true_relevance=None, queryID=None, docIDs=None):
    """
    model: trained model
    query: list of document features that needs to be ranked (only single query!) SHOULD BE PREPROCESSED
    """
    if true_relevance is not None:
        print("NDCG score", ndcg_scorer(model, query, true_relevance))
    if docIDs is None:
        docIDs = [None] * len(query)
    if type(queryID) != str:
        queryID = queryID[0]
    # ranking documents
    # print(query, query.values)
    score = model.predict(query.values)

    ranks = pd.DataFrame({'QueryID': [queryID] * len(score), 'Docid': docIDs, 'Score': score})

    ranks.sort_values(by='Score', ascending=False, inplace=True)

    return ranks.reset_index(drop=True)


if __name__ == "__main__":
    import sys

    print(sys.argv)
    sweep = True if len(sys.argv) > 1 and str(sys.argv[1]) == 'sweep' else False
    filename = 'finalized_model.sav'

    try:
        _ = pd.read_csv('queries_vec.csv').set_index('qid')
    except:
        break_document_into_chunks()

        docid2chunk = pd.read_csv('docs_chunked/docid2chunk.csv')

        # Documents:
        # - Lowercase the text
        # - Expand Contractions
        # - Clean the text
        # - Remove Stopwords
        # - Lemmatize words
        #
        # Queries:
        # - Lowercase the text
        # - Expand Contractions
        # - Clean the text

        preprocess_all_chunks(docid2chunk)
        create_w2v_model()
        w2v_model = KeyedVectors.load_word2vec_format('w2v_model.bin', binary=True)
        convert_queries_and_doc_chunks_to_vectors(docid2chunk, w2v_model)

    # load vectored docs and queries
    print("Loading vector data")
    vqueries_df = pd.read_csv('queries_vec.csv').set_index('qid')
    vdocs_df = pd.read_csv('documents_vec.csv').set_index('Docid')
    vqueries_df['vector'] = vqueries_df['vector'].map(lambda x: np.fromstring(x[1:-1], sep=' '))
    vdocs_df['vector'] = vdocs_df['vector'].map(lambda x: np.fromstring(x[1:-1], sep=' '))

    if sweep:
        w2v_model = KeyedVectors.load_word2vec_format('w2v_model.bin', binary=True)
        print('Vocabulary size:', len(w2v_model.key_to_index))

        # load training and testing data
        print("Loading training data")
        train_df = pd.read_table('train.tsv')
        test_df = pd.read_table('test.tsv')

        # find cosine similarity
        print("Finding cosine similarity")
        train_query_vec = vqueries_df.loc[train_df['#QueryID']]
        test_query_vec = vqueries_df.loc[test_df['#QueryID']]
        train_doc_vec = vdocs_df.loc[train_df['Docid']]
        test_doc_vec = vdocs_df.loc[test_df['Docid']]

        train_df['cosine_sim'] = [cosine_similarity(train_query_vec.iloc[i][0], train_doc_vec.iloc[i][0]) for i in
                                  range(len(train_df))]
        test_df['cosine_sim'] = [cosine_similarity(test_query_vec.iloc[i][0], test_doc_vec.iloc[i][0]) for i in
                                 range(len(test_df))]

        # ## Splitting and preparing data
        # validation and training split
        split = int(len(train_df) * 0.8)
        vali_df = train_df.iloc[split:]
        train_df = train_df.iloc[:split]

        # grouping according to queries and giving group numbers

        train_queries = train_df['#QueryID'].unique()
        query_tr = pd.DataFrame({'#QueryID': train_queries, 'group': range(len(train_queries))}).set_index("#QueryID")
        X_train = train_df.drop(['#QueryID', 'Label', 'Docid'], axis=1)
        y_train = train_df["Label"]

        vali_queries = vali_df['#QueryID'].unique()
        query_vali = pd.DataFrame({'#QueryID': vali_queries, 'group': range(len(vali_queries))}).set_index("#QueryID")
        X_vali = vali_df.drop(['#QueryID', 'Label', 'Docid'], axis=1)
        y_vali = vali_df["Label"]

        test_queries = test_df['#QueryID'].unique()
        query_test = pd.DataFrame({'#QueryID': test_queries, 'group': range(len(test_queries))}).set_index("#QueryID")
        X_test = test_df.drop(['#QueryID', 'Docid'], axis=1)

        # normalizing data
        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_test = feature_scaler.transform(X_test)
        X_vali = feature_scaler.transform(X_vali)

        # get the training model
        model, param_grid = get_model_and_paramgrid()

        # assign group number for query training data and create K-fold CV setup
        flatted_group = np.array([query_tr.loc[x, 'group'] for x in train_df['#QueryID']])
        gkf = GroupKFold(n_splits=5)
        cv = gkf.split(X_train, y_train, groups=flatted_group)
        cv_group = gkf.split(X_train, groups=flatted_group)  # separate CV generator for manual splitting groups

        # # generator produces `group` argument for each fold
        # def group_gen(flatted_group, cv):
        #     for train, _ in cv:
        #         print(flatted_group[train])
        #         yield np.unique(flatted_group[train], return_counts=True)[1]
        # gen = group_gen(flatted_group, cv_group)

        # create Grid search

        ndcg_scorer = make_scorer(NDCG_loss_func, greater_is_better=True)
        # gd_sr = RandomizedSearchCV(estimator=model,
        #                            param_distributions=params_grid,
        #                            n_iter=100,
        #                            cv=cv,
        #                            verbose=2,
        #                            scoring=ndcg_scorer,
        #                            refit=False)
        # gd_sr.fit(X_train, y_train)

        gd_sr = GridSearchCV(estimator=model,
                             param_grid=param_grid,
                             scoring=ndcg_scorer,
                             cv=cv,
                             n_jobs=-1)
        # fitting data
        print("Fitting data")
        gd_sr.fit(X_train, y_train)

        best_parameters = gd_sr.best_params_
        best_result = gd_sr.best_score_
        model = gd_sr.best_estimator_
        # save the model to disk
        print("Saving model")
        pickle.dump(model, open(filename, 'wb'))

        print(best_parameters)
        print(best_result)
        print(pd.DataFrame(gd_sr.cv_results_))
    else:

        # inference from the trained model
        print("Loading model")
        model = pickle.load(open(filename, 'rb'))  # load

    # load test data
    print("Loading inference data")
    test_df = pd.read_table('test.tsv')  

    # find cosine similarity
    print("Calculating cosine similarity")
    test_query_vec = vqueries_df.loc[test_df['#QueryID']]
    test_doc_vec = vdocs_df.loc[test_df['Docid']]
    test_df['cosine_sim'] = [cosine_similarity(test_query_vec.iloc[i][0], test_doc_vec.iloc[i][0]) for i in
                             range(len(test_df))]

    test_queries = test_df['#QueryID'].unique()
    query_test = pd.DataFrame({'#QueryID': test_queries, 'group': range(len(test_queries))}).set_index("#QueryID")
    X_test = test_df.drop(['#QueryID', 'Docid'], axis=1)

    idx = np.array([query_test.loc[x, 'group'] for x in test_df['#QueryID']])
    result = pd.DataFrame(columns=['QueryID', 'Docid', 'Score'])
    for i in range(len(test_queries)):
        iresult = ranking_ir(model, X_test[idx == i], None, test_df.loc[idx == i, '#QueryID'].unique(),
                             test_df.loc[idx == i, 'Docid'])
        result = pd.concat([result, iresult])
    result = result.reset_index()
    result = result[['QueryID', 'Docid', 'Score']]
    result.to_csv('A2.run', sep='\t', index=False, header=None)  #save tsv 
    print(result)
