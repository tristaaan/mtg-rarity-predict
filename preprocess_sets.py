import glob
import json
import re

import pandas as pd

from os import path


class mCard(object):
    def __init__(self, json_card):
        self.name     = json_card['name']
        self.rarity   = json_card['rarity']
        self.text     = json_card['text']
        self.original_text = json_card['original_text']
        self.supertypes = json_card['supertypes']
        self.types     = json_card['types']
        self.subtypes  = json_card['subtypes']
        self.cmc       = json_card['cmc']
        self.mana_cost = json_card['mana_cost']
        self.image_url = json_card['image_url']
        self.printings = json_card['printings']


def strip_text(card):
    if card.text == None:
        return ''
    # cast to lowercase
    val = card.text.lower()
    # remove parenthetical explanations
    ret = re.sub(r'\(.*\)', '', val)
    # remove punctuation
    ret = re.sub(r'</?i>|[,.:—•"\'\u2212]', '', ret)
    # replace \n with ' '
    ret = re.sub(r'\n', ' ', ret)
    # remove reference to self
    ret = ret.replace(card.name.lower(), 'this')
    # replace tap icon with the word
    ret = ret.replace('{t}', 'tap')
    # replace or remove counters 1/1, +1/+1, -1/+1, etc
    if re.search(r'[+-]?[\dx]+\/[+-]?[\dx]+', ret) is not None:
        ret = counter_replace(ret)
    if '{' in ret:
        # find mana groups
        mana_groups = re.finditer(r'(\{[\w\d]+(?:\/\w)?\})+', ret)
        # computed iterable to list
        mana_groups = [g.group() for g in mana_groups]
        # sort longest to shortest, this is because {4}{r} produces two groups:
        # {4}{r} and {r}, the longest one should be addressed first
        mana_groups.sort(key=lambda x: len(x), reverse=True)
        for g in mana_groups:
            # the group might have been removed if it was apart of a larger one
            if g in ret:
                ret = ret.replace(g, cost_to_cmc(g))
                # sometimes "pay" is already there
                ret = ret.replace('pay pay', 'pay')
                # don't override adding mana
                ret = ret.replace('add pay', 'add')
    return ret


def join_type(types, subtypes):
    if types[0] == 'Tribal':
        joined = ' '.join(types[1:]).lower()
    else:
        joined = ' '.join(types).lower()
    # do not consider these types
    if 'land' in joined or \
        'gate' in joined or \
        'planeswalker' in joined or \
        'Saga' in subtypes:
        return None
    return joined


def counter_replace(ability):
    # +1/+1
    ability = replace_instances(ability, r'(\+[\dx]+/\+[\dx]+)', 'enhance')
    # -1/-1
    ability = replace_instances(ability, r'(-[\dx]+/-[\dx]+)', 'weaken')
    # +1/-1
    ability = replace_instances(ability, r'(\+[\dx]+/-[\dx]+)', 'strengthen')
    # -1/+1
    ability = replace_instances(ability, r'(-[\dx]+/\+[\dx]+)', 'toughen')
    # x/x token
    ability = replace_instances(ability, r'([\dx]+/[\dx]+)', '')
    return ability


def replace_instances(ability, regex, replacement):
    groups = re.finditer(regex, ability)
    for g in groups:
        ability = ability.replace(g.group(), replacement)
    return ability


def cost_to_cmc(cost):
    cost_matches = re.findall(r'\{([\w\d]+(?:\/\w)?)\}', cost)
    energy = False
    cmc_num = 0
    cmc_str = ['pay']
    for m in cost_matches:
        # colorless always comes first
        if m in '123456789':
            cmc_num += int(m)
        # cost can be 0
        elif m == '0':
            cmc_str.append('nothing')
        # get energy
        elif m == 'e':
            cmc_num += 1
            energy = True
        # x
        elif m == 'x':
            cmc_str.append('something')
        # it's a mana letter
        else:
            cmc_num += 1
    # energy case
    if energy:
        return str(cmc_num) + ' energy'
    # 'pay something and 4'
    elif len(cmc_str) > 1 and cmc_num > 0:
        return ' '.join(cmc_str) + ' and ' + str(cmc_num)
    # 'pay 4'
    elif len(cmc_str) == 1 and cmc_num > 0:
        return ' '.join(cmc_str) + ' ' + str(cmc_num)
    # 'pay nothing'
    return ' '.join(cmc_str)

'''
colorless, red, blue, black, green, white, x
an their combinations
'''
mana_types = ['C', 'R', 'U', 'B', 'G', 'W', 'X', \
    'B/G', 'B/R', 'G/U', 'G/W', 'R/G', 'R/W', 'U/B', 'U/R', 'W/B', 'W/U']


def mana_cost_to_dict(cost):
    '''
    convert the mana cost into a dictionary with all other costs
    '''
    cost_dict = dict(zip(mana_types, [0]*len(mana_types)))
    cost_matches = re.findall(r'\{(\d+|\w\/\w|\w)\}', cost)
    for c in cost_matches:
        # some cards in NLP have this type
        if 'P' in c:
            cost_dict[c[0]] += 1
        # colorless
        elif c[0] in '1234567890':
            cost_dict['C'] += int(c)
        elif c[0] == 'X':
            cost_dict['X'] += 1
        else:
            cost_dict[c] += 1
    return cost_dict


def process_set(card_set, set_name, duplicates, df):
    print('preprocessing %s' % set_name, end='\r')
    added_cards = 0
    for jc in card_set:
        # if jc['image_url'] == None:
        #     continue
        c = mCard(jc)

        joined_type = join_type(c.types, c.subtypes)
        description = strip_text(c)
        # skip if: non-allowed type, no description, or no mana cost.
        if joined_type == None or \
            len(description) == 0 or \
            c.mana_cost == None or \
            c.name in duplicates.keys():
            continue

        if len(c.printings) > 1:
            duplicates[c.name] = c.printings

        df = df.append({'set': set_name, 'name': c.name,
                       'rarity': c.rarity.lower(),
                       'text': description, 'type': joined_type,
                       'legendary': ('Legendary' in c.supertypes) or \
                                    ('Tribal' in c.types),
                       'image_url': c.image_url,
                       'cmc': c.cmc, **mana_cost_to_dict(c.mana_cost)},
                       ignore_index=True)
        added_cards += 1
    if added_cards == 0:
        print('-- No cards added for set %s!' % set_name)
    else:
        print('processed %s, %d added' % (set_name, added_cards))
    return df, duplicates



if __name__ == '__main__':
    # get a list of the json files
    files = glob.glob(path.join('sets', '*.json'))
    columns = ['name', 'rarity', 'text', 'type', 'cmc'] + mana_types

    # initialize dataframe and iterate through files
    df = pd.DataFrame(columns=columns)
    duplicates = {}
    for fname in files:
        set_name = path.basename(fname).split('.')[0]
        with open(fname, 'r') as card_set_f:
            card_set = json.load(card_set_f)
            df, duplicates = process_set(card_set, set_name, duplicates, df)

    rare_counts = df['rarity'].value_counts(dropna=False)
    print('rarities:', rare_counts)
    type_counts = df['type'].value_counts(dropna=False)
    print('types:', type_counts)
    df.to_csv('processed_sets.csv', sep='\t')
