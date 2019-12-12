This is my solution to the [kaggle quora question pair](https://www.kaggle.com/c/quora-question-pairs) challenge! (scored top 6%!)

Note: Not all this code is mine. Kaggle competitions can be very collaborative and it is common for people to borrow features/whatever from various kernels to incorporate into their models. MOST of the code is referenced appropriately, but I could have missed a few things.

The general strategy I did in the competition was to build/collect as many applicable features as possible and then throw it into xgboost. Not ideal, but I found that I got the most gains feature engineering, rather than ensembling (I think this is the norm)

File descriptions:

The submission consists of several scripts that generate features from the quora data set. Each FeaturesX.py file will load the input data, and previously generated files, and then subsequently at a new set of engineered features.

The generated features are then saved in .Pickle files. If I where to do this again, I would use a different format for saving files to disk. Although pickling is convenient, saving larger datasets chews up memory.

I recommend 32GB of system memory when if running these scripts.

File descriptions

feat_gen.py: some cleaning functions, plus magic features
Features2.py: Build n-gram features. Mostly set type features applied to generated n-grams for each question

Features3.py: More n-gram features. This time using characters instead of words

Features4.py: Bag of words features. This was a fun feature to build. Essentially takes all the n-gram features
Generated previously, represents everything via a countvectorizor, and then does a naive bayes embedding (cross-validated out of fold prediction). I got some large gains out of this.

Features5.py: Fuzzy matching features and GoogleNews vector embedding distances, from Abhishek Thakur (https://github.com/abhishekkrthakur)

Features6.py: Magic feature. From kaggle kernel.

Features7_3.py: Graph theory features. These are 'magic' features. Many people came up with similar solutions on kaggle, this is mine.

Features7_3_process.py: Process script to load Features7_3.py results properly

Features8.py: tfidf weighted features. From kaggle kernel.

Features9.py Glove embedding. Similar to Features5.py. Basically you just use one of these fancy word2vec embeddings and then throw a bunch of distance metrics at the vectors.

SingleXGB.py: Just a single XGBoost model! In this competition most gains were from feature engineering. Plus, building an good ensemble takes a great deal of computer juice.


