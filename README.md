# News-articles-clustering
Fuzzy clustering (using fuzzy C-means algorithm) system for news articles, with Information Retrieval query system

This work was done as part of a postgraduate course that I credited during Spring 2016 at BITS Pilani, India. Fuzzy C-means is a clustering algorithm based on K-means clustering, but different in that it assigns probabilistic membership of a document to multiple clusters. This repo contains code to fuzzy cluster Times of India news articles, and then retrieve top n documents similar to a queried document. In order to overcome the dependency of clustering efficiency on initial assignment of cluster centers, an evolutionary algorithm called Harmony Search is used to arrive at good initial cluster assignments prior to the clustering. 

## Usage:

1) Read the paper (Paper.pdf) for more understanding
2) The directory 'toi' has datasets crawled from the Times of India website
3) Maintain the directory structure of the repo and run admfinal.py (look at comments in the code for better understanding of output format, code sections etc.
4) test.py is a short code we used while figuring out how to compute cosine_similarity scores for all pairs from a set of documents
5) Results are in our project report (Report.pdf)
