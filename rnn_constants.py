from os import path

MAXLEN = 74 # cut text after this many words
MAXFEAT = 1730 # maximum features in embedding

RARITIES = ['common', 'uncommon', 'rare', 'mythic']
FULL_INPUTS = ['type', 'C', 'R', 'U', 'B', 'G', 'W', 'X',  \
  'B/G', 'B/R', 'G/U', 'G/W', 'R/G', 'R/W', 'U/B', \
  'U/R', 'W/B', 'W/U', 'legendary']

DEFAULT_EMBEDDING = path.join('tmp', 'embedding_matrix.npy')
DEFAULT_TOKENIZER = path.join('tmp', 'default_tokenizer.pickle')
DEFAULT_WEIGHTS = path.join('tmp', 'weights-rnn.hdf5')
