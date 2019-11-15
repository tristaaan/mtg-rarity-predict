import json
import argparse

from time import time
from os import path
from mtgsdk import Card

from utils import make_folder

def card_to_json(c):
    c = c.__dict__
    del c['foreign_names']
    return c

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch MGT sets')
    parser.add_argument('-start', '-s', help='the set to start at',
                        default='M10')
    args = parser.parse_args()
    kw = vars(args)
    start_at = kw['start']

    # Expansion and core sets back until October 2015
    # https://mtg.gamepedia.com/Core_set
    # https://mtg.gamepedia.com/Set#List_of_Magic_expansions_and_sets
    sets = ['M10', 'ZEN', 'WWK', 'ROE',
            'M11', 'SOM', 'MBS', 'NPH',
            'M12', 'ISD', 'DKA', 'AVR',
            'M13', 'RTR', 'GTC', 'DGM',
            'M14', 'THS', 'BNG', 'JOU',
            'M15', 'KTK', 'FRF', 'DTK',
            'ORI', 'BFZ', 'OGW', 'SOI',
            'EMN', 'KLD', 'AER', 'AKH',
            'HOU', 'XLN', 'RIX', 'DOM',
            'M19', 'GRN', 'RNA', 'WAR',
            'M20', 'ELD']
    start_index = sets.index(start_at.upper())
    dir_name = 'sets'
    make_folder(dir_name)
    for s in sets[start_index:]:
        print('fetching %s...' % s)
        start = time()
        this_set = Card.where(set=s).all()
        current_set = []
        for c in this_set:
            current_set.append(card_to_json(c))
        with open(path.join(dir_name, '%s.json' % s), 'w') as outfile:
            json.dump(current_set, outfile)
        print('done %s in %0.2fs' % (s, time() - start))
    print('done')
