# Building Next-item recommendation system on top of Amazon's explicit ratings dataset, using deep learning with Keras
The Next-item recommendation challenge is to recommend top-k potentially preferred items for the target user, based on the user’s explicit rating or implicit behaviors over items.

Explicit feedback is when users gives voluntarily the rating information on what they like and dislike.
In our case, we have explicit item ratings ranging from one to five.

I framed the recommendation system as a rating prediction machine learning challenge and created a hybrid architecture that combines the collaborative filtering and content-based approaches:
- Collaborative: Learn item similarities from user's interaction and recommend to the user items which he is likely to rate high according to learnt item & user embeddings.
- Content based: Learn item similarities from item’s metadata attributes (such as price and title), and recommend to the user contents similar to those he rated high.

I created and compared 2 explicit recommendation engines for predicting user's ratings  based on 2 machine learning architecture: 
- Matrix Factorization: Perform a dot product between the respective user and item embeddings.
- Deep neural network: Merge user and item embeddings by concatenation or multiplication, and then use them as features for the neural network.
For training, validation and prediction, I used the electronics reviews dataset from Amazon for the period May 1996-July 2014, which contains explicit item ranking and in which all users and items have at least 5 reviews, and compare the results to using all the samples from the electronics reviews dataset of amazon (regardless of reviews number).
I compared the results of matrix factorization and different configurations of neural networks on the 2 datasets in order to find a "best" model for predicting the recommended items

The 2 datasets can be downloaded from: http://jmcauley.ucsd.edu/data/amazon/links.html . Acknowledgment to Julian McAuley for the links.

At the end, I used the best models to recommend items to users. 



