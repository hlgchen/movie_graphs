import ast
import itertools
from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import networkx as nx



################### meta features ###################

def prep_adult(data):
    data.fillna(data.mode()[0], inplace=True)
    d = {'False':0,'True':1}
    return data.map(d)

def prep_video(data):
    data.fillna(data.mode()[0], inplace=True)
    d = {False:0,True:1}
    return data.map(d)

def prep_budget(data):
    return np.log(data.astype(float)+1)

def prep_rev(data):
    return np.log(data.astype(float)+1)

def prep_votes(data):
    return np.log(data.astype(float)+1)

def prep_rating(data):
    return data.astype(float)/10

def prep_runtime(data):
    return data.astype(float)/100

def prep_lang(data, movie_data):
    id_lang = movie_data['original_language'].drop_duplicates().reset_index().to_dict()['original_language']
    lang_id = {v: k for k,v in id_lang.items()}
    return data.map(lang_id).astype(int)

def prep_status(data, movie_data):
    id_status = movie_data['status'].drop_duplicates().reset_index().to_dict()['status']
    lang_id = {v: k for k,v in id_status.items()}
    return data.map(lang_id).astype(int)

def prep_genres(data):
    # one hot encode the genres 
    genres = data.apply(ast.literal_eval)
    genres = genres.apply(lambda row: [d["name"] for d in row])

    mlb = MultiLabelBinarizer()
    one_hot_genres = pd.DataFrame(
        mlb.fit_transform(genres), columns=mlb.classes_, index=genres.index
    )
    one_hot_genres.columns = one_hot_genres.columns.str.lower()
    one_hot_genres.columns = one_hot_genres.columns.str.replace(" ", "_")
    one_hot_genres.columns = "genre_" + one_hot_genres.columns
    
    return one_hot_genres


################### credits features ###################


def preprocess_credits(credits):
    credits = credits.copy()
    credits.cast = credits.cast.apply(ast.literal_eval)
    credits.crew = credits.crew.apply(ast.literal_eval)
    cdf = credits[["tmdbid"]].copy()
    cdf["cast_list"] = credits.cast.apply(lambda x: [y["name"] for y in x])
    cdf["crew_list"] = credits.crew.apply(lambda x: [y["name"] for y in x])
    cdf["people"] = cdf.apply(lambda x: x.cast_list + x.crew_list, axis=1)
    return cdf.drop(columns=["cast_list", "crew_list"])


def get_graph(_s, map_names=False):
    """Takes pd.Series containing lists of people who have worked together.
    Returns graph showing cooperation between these people
    """

    s = _s.copy()

    nodes = s.explode().drop_duplicates().reset_index(drop=True)

    if map_names:
        nodes_inverse_mapping = {v: k for k, v in nodes.to_dict().items()}
        s = s.apply(lambda x: [nodes_inverse_mapping[name] for name in x])

    edges_raw = s.apply(
        lambda x: [tuple(sorted(edge)) for edge in itertools.combinations(x, 2)]
    )
    edges_raw = edges_raw.explode().value_counts().reset_index()
    edges_raw.columns = ["edge", "weight"]
    _a = np.array(edges_raw.edge.tolist(), dtype="object")
    _b = np.array(edges_raw.weight.tolist(), dtype="object").reshape(-1, 1)
    _data = np.concatenate([_a, _b], axis=1)
    edges_weighted = [tuple(_data[i]) for i in range(len(_data))]
    G = nx.Graph(directed=False)
    G.add_nodes_from(nodes.to_list())
    G.add_weighted_edges_from(edges_weighted)

    return G


def get_high_degree_people(G, n):
    degree_dict = dict(G.degree(G.nodes()))
    return sorted(degree_dict.items(), key=itemgetter(1), reverse=True)[:n]

def get_pagerank(G, n):
    pr = nx.pagerank(G, weight = 'weight')
    return sorted(pr.items(), key=lambda item: item[1], reverse=True)[:n]


def prep_connected_people(data):
    # one hot encode the connected people 
    
    mlb = MultiLabelBinarizer()
    one_hot_people = pd.DataFrame(
        mlb.fit_transform(data), columns=mlb.classes_, index=data.index
    )
    one_hot_people.columns = one_hot_people.columns.str.lower()
    one_hot_people.columns = one_hot_people.columns.str.replace(" ", "_")
    one_hot_people.columns = "people_" + one_hot_people.columns
    
    return one_hot_people