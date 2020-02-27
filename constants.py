from os import path

MAXLEN = 74 # cut text after this many words
MAXFEAT = 1730 # maximum features in embedding

RARITIES = ['common', 'uncommon', 'rare', 'mythic']
FULL_INPUTS = ['type', 'C', 'R', 'U', 'B', 'G', 'W', 'X',  \
  'B/G', 'B/R', 'G/U', 'G/W', 'R/G', 'R/W', 'U/B', \
  'U/R', 'W/B', 'W/U', 'legendary']

# Expansion and core sets back until October 2015
# https://mtg.gamepedia.com/Core_set
# https://mtg.gamepedia.com/Set#List_of_Magic_expansions_and_sets
SETS = ['M10', 'ZEN', 'WWK', 'ROE',
        'M11', 'SOM', 'MBS', 'NPH',
        'M12', 'ISD', 'DKA', 'AVR',
        'M13', 'RTR', 'GTC', 'DGM',
        'M14', 'THS', 'BNG', 'JOU',
        'M15', 'KTK', 'FRF', 'DTK',
        'ORI', 'BFZ', 'OGW', 'SOI',
        'EMN', 'KLD', 'AER', 'AKH',
        'HOU', 'XLN', 'RIX', 'DOM',
        'M19', 'GRN', 'RNA', 'WAR',
        'M20', 'ELD', 'THB']

GLOVE_DIR = '.' # directory for pre-trained Glove 6B files

DEFAULT_EMBEDDING = path.join('tmp', 'default_embedding_zzz.npy')
DEFAULT_TOKENIZER = path.join('tmp', 'default_tokenizer.pickle')
DEFAULT_WEIGHTS = path.join('tmp', 'weights-rnn.hdf5')
