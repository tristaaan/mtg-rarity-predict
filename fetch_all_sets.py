import json
from time import time
from mtgsdk import Card

def card_to_json(c):
    c = c.__dict__
    del c['foreign_names']
    return c

if __name__ == '__main__':
    # Expansion and core sets back until October 2015
    # https://mtg.gamepedia.com/Core_set
    # https://mtg.gamepedia.com/Set#List_of_Magic_expansions_and_sets
    sets = ['BFZ', 'OGW', 'SOI', 'EMN', # no core set, Magic Origins (ORI)
            'KLD', 'AER', 'AKH', 'HOU', # no core set
            'XLN', 'RIX', 'DOM', 'M19',
            'GRN', 'RNA', 'WAR', 'M20',
            'ELD'] #_     _       M21
    processed_sets = {}
    for s in sets:
        start = time()
        this_set = Card.where(set=s).all()
        processed_sets[s] = []
        for c in this_set:
            processed_sets[s].append(card_to_json(c))
        print('processed %s in %0.2fs' % (s, time() - start))

    with open('all_sets.json', 'w') as outfile:
      json.dump(processed_sets, outfile)
