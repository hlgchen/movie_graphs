import torch
import numpy as np

def recommend_for(user, movies, model):
    """Returns ranked movies and the corresponding predictions by the model, 
    for a given user."""
    u_node = torch.full(movies.shape, user)
    edges = torch.cat([u_node, movies])
    ranked_edges, pred = model.recommend(edges)
    return ranked_edges, pred

def get_watched_movies(user, train_edges):
    """Returns movies that user has already watched 
    (i.e. an edge with user exists in train_edges).
    Return type is one dimensional tensor.
    """
    mask1 = train_edges[0, :] == user
    mask2 = train_edges[1, :] == user
    return torch.cat([train_edges[0, mask2], train_edges[1, mask1]]).tolist()


def get_unwatched_movies(user, train_edges, library):
    """Returns one dimensional tensor of movies that are in the library, 
    but that haven't been watched by user. 
    I.e. the movies are in the library and do not appear together with user 
    in train_edges. 
    """
    watched = set(get_watched_movies(user, train_edges))
    library = set(list(library))
    return torch.tensor(list(library - watched))


def recall_at_k(user, train_edges, test_edges, model, library, k=50):
    """Return recall@k value for a given user. 
    What percentage of movies that user watched in the test_edges, 
    are among the top k recommendations made by the model. 
    Top k recommendations do not include movies that have already been watched
    (i.e. contained in train_edges.)
    """
    unwatched = get_unwatched_movies(user, train_edges, library)
    unwatched = unwatched.unsqueeze(0)
    ranked_edges, _ = recommend_for(user, unwatched, model)
    recommendations = set(ranked_edges[1, :k].tolist())
    watched_test = set(get_watched_movies(user, test_edges))

    hit = recommendations.intersection(watched_test)

    return len(hit) / len(watched_test)


def avg_recall_at_k(train_edges, test_edges, model, library, users,k=50):
    """Calculates average over recall_at_k for all users appearing in test_edges."""
    
    users_in_test_edges = set(torch.flatten(test_edges).tolist())
    target_users = set(list(users)).intersection(users_in_test_edges)
    
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