# MTG Rarity Predict

Predict Magic: The Gathering card rarity.

- `fetch_sets.py`: fetches all core and expansion sets since M10. This can take some time and the service can timeout. There is a `-s [set code]` argument which starts the download at the given set.
- `preprocess_sets.py`: processes all the downloaded sets in the `/sets` folder. This normalizes the text in card abilities and removed unwanted card types (lands, planeswalkers, saga enchantments). The output of this process is `preprocessed_sets.csv`
- `naive_prediction.py`: Uses a simple model from the SKLearn library to predict rarity. Takes the argument `-c` for classifier which is one of "mlp" (multi-layer perceptron), "svm" (support vector machine), and "rf" (random forest). Default is "mlp". 
- `rnn_model.py`: The RNN model
- `rnn_prediction.py`: Trains and tests the RNN model
- `rnn_viz.py`: Visualizes the output of an RNN model as a confusion matrix.
- `utils.py`: Various utilities used in most of the files above. Primary functions are retrieving a train/test data split from a `.csv` file and creating the word embedding matrix. 