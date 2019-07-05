import json
from time import time
from mtgsdk import Card

def card_to_json(c):
    c = c.__dict__
    del c['foreign_names']
    return c

if __name__ == '__main__':
    sets = ['MH1', 'WAR', 'RNA', 'UMA', 'GRN',
            'M19', 'DOM', 'RIX', 'XLN', 'M20']
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
