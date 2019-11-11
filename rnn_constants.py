MAXLEN = 84 # cut text after this many words
MAXFEAT = 1024 # maximum features in embedding

RARITIES = ['common', 'uncommon', 'rare', 'mythic']
FULL_INPUTS = ['type', 'C', 'R', 'U', 'B', 'G', 'W', 'X',  \
  'B/G', 'B/R', 'G/U', 'G/W', 'R/G', 'R/W', 'U/B', \
  'U/R', 'W/B', 'W/U', 'legendary']