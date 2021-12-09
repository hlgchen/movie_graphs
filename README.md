# movie_recommendations_cs224w

This is the classproject in Stanford CS224w of 

- samzliu
- hlgchen

We implement an embedding model with subsequent LGCN embedding smoothing for movie recommendations (The Movies Dataset on Kaggle). 
Technical ideas can be read in the following blogpost: 
.....

To run the main colab (colab/movie_recommendations.ipynb) click on the icon. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hlgchen/movie_graphs/blob/main/colab/movie_recommendations.ipynb)

Repo structure: 
- colab: contains the main IPython notebook with the implementation of the graphical movie recommender 
- code: contains implementation of movie recommender and extensions. Different to the main notebook in the colab folder. Code here is more modular. 
- data: should contain csv datafile from the movies dataset, can be downloaded using the notebook
