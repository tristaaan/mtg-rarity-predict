# MTG Rarity Predict

Predict Magic: The Gathering card rarity.

- `fetch_sets.py`: fetches all core and expansion sets since M10. This can take some time and the service can timeout. There is a `-s [set code]` argument which starts the download at the given set.
- `preprocess_sets.py`: processes all the downloaded sets in the `/sets` folder. This normalizes the text in card abilities and removed unwanted card types (lands, planeswalkers, saga enchantments). The output of this process is `preprocessed_sets.csv`
- `naive_prediction.py`: Uses a simple model from the SKLearn library to predict rarity. Takes the argument `-c` for classifier which is one of "mlp" (multi-layer perceptron), "svm" (support vector machine), and "rf" (random forest). Default is "mlp". 
- `rnn_model.py`: The RNN model
- `rnn_prediction.py`: Trains and tests the RNN model. Arguments:
    + `-m` model variant, one of 'lstm' or 'conv'. Default 'lstm'.
    + `-e` path for embedding file. Default is 'tmp/embedding_matrix.npy', will be built from 'Glove.6B.200d.txt' if it doesn't exist.
    + `-t` path for tokenizer file. Default is 'tmp/default_tokenizer.pickle', will be built with default embedding if it doesn't exist.
- `rnn_viz.py`: Visualizes the output of an RNN model as a confusion matrix. Arguments are same as in `rnn_prediction.py` however embedding and tokenizer files will not be built if they do not exist.
- `utils.py`: Various utilities used in most of the files above. Primary functions are retrieving a train/test data split from a `.csv` file, creating the word embedding matrix, and creating confusion matrices. 