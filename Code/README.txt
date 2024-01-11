This repository contains all the code required to produce the document embeddings and use them for article similarity matching + k-means clustering.

Classes folder contains files with class objects designed to make handling the data simpler.

1.Train Embeddings.py trains the document embedding models and saves the article vectors to the "Embeddings" folder in the Data directory.

2a. Similarity overlap.py runs a loop for each set of embeddings that compares similarities of articles against others of their class, the class types being root number and version. It outputs the boxplots that show the overlap between root similarities and version similarities. The purpose is to demonstrate that they are not entirely separate.

2b. Assignment by similarity.py runs a loop on each set of embeddings that runs the code in match_articles.py. This code loops over the articles in the data set and finds the most similar article in the previous version of the Standing Orders. The assumption is that if the article is not an entirely new article, the matched article should have the same root number. This code forms the basis of the proposed "half-way solution" mentioned in the paper, but building the infrastructure to make that work was out of scope for this project.

3a. All kmeans.py runs all of the k-means models and saves the clusters to the results database.

3b. Evaluate kmeans.py loops over the results DB table with the class labels and compares them to the known root labels. Outputs the accuracy of each model in a table.

waiter.py just contains some functions that were helpful for making sure the timers were working correctly
