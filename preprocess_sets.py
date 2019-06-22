import json
import re
import pandas as pd


class mCard(object):
    def __init__(self, json_card):
        self.name = json_card['name']
        self.rarity = json_card['rarity']
        self.original_text = json_card['original_text']
        self.types = json_card['types']
        self.cmc  = json_card['cmc']


def strip_text(val):
    if val == None:
        return ''
    ret = re.sub(r' +', ' ', re.sub(r'</?i>|[\(\),./:\n—•]', ' ', val.lower()))
    if '{' in ret:
        mana_groups = re.finditer(r'(\{[\d\w]+\})+', ret)
        for g in mana_groups:
            ret = ret.replace(g.group(), cost_to_cmc(g.group()))

    return ret

def join_type(val):
    joined = ' '.join(val).lower()
    if 'land' in joined or 'gate' in joined:
        return None
    return joined

def cost_to_cmc(cost):
    cost_matches = re.findall(r'\{([\w\d]+)\}', cost)
    cmc = 0
    if cost_matches[0] in '1234567890':
        cmc += int(cost_matches[0])
    cmc += len(cost_matches[1:])
    return str(cmc)

def process_set(card_set, df):
    for jc in card_set:
        c = mCard(jc)
        joined_type = join_type(c.types)
        if joined_type == None:
            continue
        df = df.append({'name': c.name, 'rarity': c.rarity.lower(),
                       'text': strip_text(c.original_text),
                       'type': joined_type, 'cmc': c.cmc}, ignore_index=True)
    return df

if __name__ == '__main__':
    with open('all_sets.json', 'r') as all_sets:
        sets = json.load(all_sets)
        set_keys = sets.keys()
        df = pd.DataFrame(columns=['name', 'rarity', 'text', 'type', 'cmc'])
        for k in set_keys:
            df = process_set(sets[k], df)
        df.to_csv('processed_sets.csv', sep='\t')
