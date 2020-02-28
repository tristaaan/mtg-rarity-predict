import json
import argparse

from time import time
from os import path
from mtgsdk import Card

from utils import make_folder
from constants import SETS as sets


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

    start_index = sets.index(start_at.upper())
    dir_name = 'sets'
    make_folder(dir_name)
    for s in sets[start_index:]:
        print('fetching %s...' % s, end='\r')
        start = time()
        this_set = Card.where(set=s).all()
        current_set = []
        for c in this_set:
            current_set.append(card_to_json(c))
        with open(path.join(dir_name, '%s.json' % s), 'w') as outfile:
            json.dump(current_set, outfile)
        print('fetched %s in %0.2fs' % (s, time() - start))
    print('done')
