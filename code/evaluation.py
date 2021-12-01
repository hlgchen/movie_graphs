import torch
import numpy as np

def recommend_for(user, movies, model):
    """
    Returns ranked movies and the corresponding predictions by the model, 
    for a given user.
    
    Params: 
        - user: integer that is the user id
        - movies: movies out of which recommendations are to be found (1, n_unwatched_movies)
        - model: model that is used to rank/evaluate the movies
        
    Returns: 
        - ranked_edges: possible edges sorted by ranking (highest score coming first)
        - pred: predicted values by the model (highest prediction coming first)
    
    """
    u_node = torch.full(movies.shape, user)
    edges = torch.cat([u_node, movies])
    ranked_edges, pred = model.recommend(edges)
    return ranked_edges, pred

def get_watched_movies(user, train_edges):
    """Returns movies that user has already watched 
    (i.e. an edge with user exists in train_edges).
    Return type is a list.
    
    params: 
        - user: integer user id
        - train_edges: torch tensor of size (2, .) with edges that were 
                used for training
    
    """
    mask1 = train_edges[0, :] == user
    mask2 = train_edges[1, :] == user
    return torch.cat([train_edges[0, mask2], train_edges[1, mask1]]).tolist()


def get_unwatched_movies(user, train_edges, library):
    """Returns one dimensional tensor of movies that are in the library, 
    but that haven't been watched by user (in train_edges). 
    I.e. the movies are in the library and do not appear together with user 
    in train_edges. 
    """
    # create set of watched movies and set of all available movies (library)
    watched = set(get_watched_movies(user, train_edges))
    library = set(list(library))
    return torch.tensor(list(library - watched))


def recall_at_k(user, train_edges, test_edges, model, library, k=50):
    """Return recall@k value for a given user. 
    What percentage of movies that user watched in the test_edges, 
    are among the top k recommendations made by the model. 
    Top k recommendations do not include movies that have already been watched
    (i.e. contained in train_edges.)
    
    Params:
        - user: int - should be the integer representing the user
        - train_edges: edges that were used during training. Shape should be (2, .)
        - test_edges: edges that are used in the test set. Shape should be (2,.)
        - model: model that makes prediction how close two nodes are to each other
                should have recommend method implemented
        - library: arraylike object with all movie ids
        - k: int, score recall at k, calculated for top k recommendations
    """
    
    unwatched = get_unwatched_movies(user, train_edges, library)
    unwatched = unwatched.unsqueeze(0) # shape (1, n_unwatched_movies)
    ranked_edges, _ = recommend_for(user, unwatched, model)
    
    # note that by construction movies are at index position 1
    recommendations = set(ranked_edges[1, :k].tolist()) 
    watched_test = set(get_watched_movies(user, test_edges))

    hit = recommendations.intersection(watched_test)

    return len(hit) / len(watched_test)


def avg_recall_at_k(train_edges, test_edges, model, library, users,k=50):
    """
    Calculates average over recall_at_k for all users and movies appearing in test_edges.
    
    Params:
        - train_edges: edges that were used during training. Shape should be (2, .)
        - test_edges: edges that are used in the test set. Shape should be (2,.)
        - model: model that makes prediction how close two nodes are to each other
                should have recommend method implemented
        - library: arraylike object with all movie ids
        - users: arraylike object with all user ids
        - k: int, score recall at k, calculated for top k recommendations
        
    Returns:
        params : float
            average recall_at_k value over all users and movies in test_edges
    """
    
    # get set of all users in test_edges
    users_movies_in_test_edges = set(torch.flatten(test_edges).tolist())
    target_users = set(list(users)).intersection(users_movies_in_test_edges)
    
    # append recall @k value for each user in test_edges to recall values
    recall_values = []
    for user in target_users:
        rec = recall_at_k(
            user=user,
            train_edges=train_edges,
            test_edges=test_edges,
            model=model,
            library=library,
            k=k,
        )
        recall_values.append(rec)
    recall_values = np.array(recall_values)
    return recall_values.mean()


def evaluate(model, label, edge):
    pred = model.forward(edge)
    return accuracy(pred, label)