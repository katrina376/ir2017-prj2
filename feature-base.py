import os
import sys

from meta import *

# argv[1]: path to training data
# argv[2]: path to testing data
# argv[3]: path to corpus (Dream_of_the_Red_Chamber_seg.txt)

PATH_TRAINING = sys.argv[1]
PATH_TESTING = sys.argv[2]
PATH_CORPUS = sys.argv[3]

CORE_POS = ['N', 'Vt']

NOUN_WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 is not None else 5
VERB_WINDOW = int(sys.argv[5]) if len(sys.argv) > 5 is not None else 5

WINDOW = {
    'N': NOUN_WINDOW,
    'Vt': VERB_WINDOW
}

PATH_OUTPUT = 'output_{}_{}.txt'.format(NOUN_WINDOW, VERB_WINDOW)

try:
    os.remove(PATH_OUTPUT)
except OSError:
    pass

print('NOUN_WINDOW:', NOUN_WINDOW)
print('VERB_WINDOW:', VERB_WINDOW)

DEFAULT_RELATION = '遠親'
RELATION_LIST = [
    '夫妻', '父女', '父子', '主僕', '兄弟姊妹', '母女', '母子', '姑叔舅姨甥侄',
    '居處', '師徒', '祖孫', '遠親']

def fetch(s, i):
    return [it[i] for it in s]

def check_exist(ep, s1, s2=None):
    """Checking if 2 entities are in the list(s) of token of given sentence(s).

    If arg `s2` is not given, the function only checks `s1` for both `e1` and
    `e2`.
    Cases only mentioning given names are considered.
    """
    e1 = ep[0]
    e2 = ep[1]
    s2 = s1 if s2 is None else s2
    t1 = fetch(s1, 0)
    t2 = fetch(s2, 0)
    return ((e1 in t1 and e2 in t2)
            or (e1[1:] in t1 and e2 in t2)
            or (e1 in t1 and e2[1:] in t2)
            or (e1[1:] in t1 and e2[1:] in t2)
            or (e1 in t2 and e2[1:] in t1)
            or (e1[1:] in t2 and e2 in t1)
            or (e1 in t2 and e2[1:] in t1)
            or (e1[1:] in t2 and e2[1:] in t1))

def search_window(pair, token_idx, context, width):
    result = False

    window = range(
        token_idx - width + 1, token_idx)

    for shift in window:
        start = 0 if shift < 0 else shift
        end = shift + width
        if check_exist(pair, context[start:end]):
            result = True

    return result

# Read corpus and eliminate non-core words
with open(PATH_CORPUS, encoding='utf8') as f:
    corpus = []
    sentence = []
    for line in f:
        token_pairs = line.strip().split(' ')
        for tpr in token_pairs:
            token = tpr.split('_')
            if token[1] in simplified_pos:
                if simplified_pos[token[1]] in CORE_POS:
                    sentence.append((token[0], token[1]))
            elif token[0] == '。':
                corpus.append(sentence)
                sentence = []

# Read training data
with open(PATH_TRAINING, encoding='utf8') as f:
    trainings = []
    line_no = 0
    for line in f:
        line_no += 1
        if line_no == 1:
            continue

        split_up = line.strip().split('\t')
        relation = split_up[3]
        entity1 = split_up[1]
        entity2 = split_up[2]

        trainings.append({
            'pair': (entity1, entity2),
            'relation': relation,
            'context': [],
            'correct': relation,
            'predict': DEFAULT_RELATION,
        })


# Read testing data
with open(PATH_TESTING, encoding='utf8') as f:
    testings = []
    line_no = 0
    for line in f:
        line_no += 1
        if line_no == 1:
            continue

        split_up = line.strip().split('\t')
        relation = split_up[3]
        entity1 = split_up[1]
        entity2 = split_up[2]

        testings.append({
            'pair': (entity1, entity2),
            'correct': relation,
            'predict': DEFAULT_RELATION,
            'context': [],
        })

# Fetch related sentences from corpus.
# In order to make the scripts breaf, zip training and testing data first.
zipups = trainings + testings

for rel in zipups:
    for idx, sentence in enumerate(corpus):
        # For each sentence in the corpus.

        # Check if both entities can be found in the sentence.
        if check_exist(rel['pair'], sentence):
            rel['context'].append(sentence)
            continue

        # Next sentence.
        # If one of the entity can be found in the next sentence,
        # add both sentences into the context of the pair.
        is_1_valid = idx < len(corpus) - 1
        if not is_1_valid:
            break

        neighbor_1_sentence = corpus[idx + 1]
        if check_exist(rel['pair'], sentence, neighbor_1_sentence):
            concatenate = [it for s in corpus[idx:idx+2] for it in s]
            rel['context'].append(sentence + neighbor_1_sentence)
            continue

        # Next sentence of the next sentence.
        # If one of the entity can be found in another sentence,
        # add all sentences (including the right next one) into the context
        # of the pair.
        is_2_valid = idx < len(corpus) - 2
        if not is_2_valid:
            break

        neighbor_2_sentence = corpus[idx + 2]
        if check_exist(rel['pair'], sentence, neighbor_2_sentence):
            concatenate = [it for s in corpus[idx:idx+3] for it in s]
            rel['context'].append(sentence + neighbor_2_sentence)
            continue

# Update the original testing and training data
trainings = zipups[:len(trainings)]
testings = zipups[len(trainings):]

# Extract features from training data
features = {}

for rel in trainings:
    pair = rel['pair']
    relation = rel['relation']
    for context in rel['context']:
        for idx, token in enumerate(context):
            for pos in CORE_POS:
                # For each POS type in the given list.

                # Check if the token POS is exactly the current POS
                if simplified_pos[token[1]] == pos:
                    # Search in the window covering the given token.
                    # Return true if the pair can be found inside the window
                    # around the token.
                    found = search_window(
                        pair, idx, context, WINDOW[pos])
                    if found:
                        # Add the token into the list of feature.
                        # Count the times when one relation has the extracted
                        # feature.
                        if token not in features:
                            # Initialize
                            features[token] = {}
                            features[token][relation] = 1
                        else:
                            if relation not in features[token]:
                                # Initialize
                                features[token][relation] = 1
                            else:
                                features[token][relation] += 1


# Check if extracted features can be found in testing pairs context.

correctness = 0
statistic = {}

for rel in RELATION_LIST:
    statistic[rel] = 0

with open(PATH_OUTPUT, 'a', encoding='UTF-8') as out:
    out.write('\t'.join(['Entity 1', 'Entity 2', 'Correct', 'Predict']))

for rel in testings:
    pair = rel['pair']
    extracts = []
    for context in rel['context']:
        for idx, token in enumerate(context):
            for pos in CORE_POS:
                if simplified_pos[token[1]] == pos:
                    found = search_window(
                        pair, idx, context, WINDOW[pos])
                    if found:
                        extracts.append(token)

    scores = {}
    for ext in extracts:
        if ext not in features:
            continue

        penalty = 1
        if simplified_pos[ext[1]] == 'N':
            penalty = float(1 / NOUN_WINDOW)
        elif simplified_pos[ext[1]] == 'Vt':
            penalty = float(1 / VERB_WINDOW)

        total = sum(count for relation, count in features[ext].items())
        for relation, count in features[ext].items():
            if relation not in scores:
                scores[relation] = penalty * float(count / total)
            else:
                scores[relation] += penalty * float(count / total)

    if len(scores) > 0:
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rel['predict'] = ranking[0][0]

    with open(PATH_OUTPUT, 'a', encoding='UTF-8') as out:
        out.write('\t'.join([
            rel['pair'][0], # Entity1
            rel['pair'][1], # Entity2
            rel['correct'], # Correct
            rel['predict']  # Predict
        ]))

    if rel['predict'] == rel['correct']:
        correctness += float(1 / len(testings))

        total = sum(
            1 for t in testings if t['correct'] == rel['correct'])
        statistic[rel['correct']] += float(1 / total)


print('CORRECTNESS:', correctness)
print('CORRECTNESS OF EACH RELATION:')
for rel in RELATION_LIST:
    print(rel, statistic[rel])
