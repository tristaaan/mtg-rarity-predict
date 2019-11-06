import json
import re
import pandas as pd


class mCard(object):
    def __init__(self, json_card):
        self.name   = json_card['name']
        self.rarity = json_card['rarity']
        self.text   = json_card['text']
        self.original_text = json_card['original_text']
        self.supertypes = json_card['supertypes']
        self.types     = json_card['types']
        self.cmc       = json_card['cmc']
        self.mana_cost = json_card['mana_cost']


def strip_text(val):
    if val == None:
        return ''
    # remove punctuation
    ret = re.sub(r'</?i>|[\(\),.:—•"\']', '', val.lower())
    # replace \n with ' '
    ret = re.sub(r' +', ' ', re.sub(r'\n', ' ', ret))
    # replace special character
    ret = ret.replace(u'\u2212', '-')
    # replace tap icon with the word
    ret = ret.replace('{t}', 'tap')
    # replace or remove counters 1/1, +1/+1, -1/+1, etc
    if re.search(r'[+-]?[\dx]+\/[+-]?[\dx]+', ret) is not None:
        ret = counter_replace(ret)
    if '{' in ret:
        mana_groups = re.finditer(r'(\{[\d\w ]+\})+', ret)
        for g in mana_groups:
            ret = ret.replace(g.group(), cost_to_cmc(g.group()))
    return ret


def join_type(val):
    joined = ' '.join(val).lower()
    # do not consider these types
    if 'land' in joined or \
        'gate' in joined or \
        'planeswalker' in joined:
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
    # 1/1
    ability = replace_instances(ability, r'([\dx]+/[\dx]+)', '')
    return ability


def replace_instances(ability, regex, replacement):
    groups = re.finditer(regex, ability)
    for g in groups:
        ability = ability.replace(g.group(), replacement)
    return ability


def cost_to_cmc(cost):
    cost_matches = re.findall(r'\{([\w\d ]+)\}', cost)
    cmc = 0
    if cost_matches[0] in '1234567890':
        cmc += int(cost_matches[0])
    cmc += len(cost_matches[1:])
    return str(cmc)

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
        if c[0] in '1234567890':
            cost_dict['C'] += int(c)
        elif c[0] == 'X':
            cost_dict['X'] += 1
        else:
            cost_dict[c] += 1
    return cost_dict


def process_set(card_set, df):
    for jc in card_set:
        if jc['image_url'] == None:
            continue
        c = mCard(jc)
        joined_type = join_type(c.types)
        description = strip_text(c.text)
        # skip if: non-allowed type, no descriptopn, or no mana cost.
        if joined_type == None or \
            len(description) == 0 or \
            c.mana_cost == None:
            continue

        df = df.append({'name': c.name, 'rarity': c.rarity.lower(),
                       'text': description, 'type': joined_type,
                       'legendary': 'Legendary' in c.supertypes,
                       'cmc': c.cmc, **mana_cost_to_dict(c.mana_cost)},
                       ignore_index=True)
    return df

if __name__ == '__main__':
    with open('all_sets.json', 'r') as all_sets:
        sets = json.load(all_sets)
        set_keys = sets.keys()
        columns = ['name', 'rarity', 'text', 'type', 'cmc'] + mana_types
        df = pd.DataFrame(columns=columns)
        for k in set_keys:
            print('preprocessing %s' % k)
            df = process_set(sets[k], df)
        df.to_csv('processed_sets.csv', sep='\t')
