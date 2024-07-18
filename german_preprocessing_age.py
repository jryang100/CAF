import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EPS = 1e-8
has_age = 1


def bucket(x, buckets):
    x = float(x)
    n = len(buckets)
    label = n
    for i in range(len(buckets)):
        if x <= buckets[i]:
            label = i
            break
    template = [0. for j in range(n + 1)]
    template[label] = 1.
    return template


def onehot(x, choices):
    if not x in choices:
        print('could not find "{}" in choices'.format(x))
        print(choices)
        raise Exception()
    label = choices.index(x)
    template = [0. for j in range(len(choices))]
    template[label] = 1.
    return template


def continuous(x):
    return [float(x)]


def parse_row(row, headers, headers_use):
    new_row_dict = {}
    for i in range(len(row)):
        x = row[i]
        hdr = headers[i]
        new_row_dict[hdr] = fns[hdr](x)
    sens_att = new_row_dict[sensitive]
    label = new_row_dict[target]
    new_row = []
    for h in headers_use:
        new_row = new_row + new_row_dict[h]
    return new_row, label, sens_att


def whiten(X, mn, std):
    mntile = np.tile(mn, (X.shape[0], 1))
    stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), EPS)
    X = X - mntile
    X = np.divide(X, stdtile)
    return X


if __name__ == '__main__':
    f_in_data = './datasets/german.csv'
    df = pd.read_csv(f_in_data)
    adult_data = df.values
    train_set, test_set = train_test_split(adult_data, test_size=0.3, random_state=42)
    hd_file = './datasets/german.headers'
    f_out_np = './datasets/german.npz'

    REMOVE_MISSING = True
    MISSING_TOKEN = '?'

    headers = 'checkingstatus1,duration,history,purpose,amount,savings,employ,installment,status,' \
              'others,residence,property,age,otherplans,housing,cards,job,liable,tele,foreign,Default'.split(',')
    if has_age:
        headers_use = 'checkingstatus1,duration,history,purpose,amount,savings,employ,installment,' \
                      'status,others,residence,property,age,otherplans,housing,cards,job,liable,tele,foreign'.split(',')
    else:
        headers_use = 'checkingstatus1,duration,history,purpose,amount,savings,employ,installment,' \
                      'status,others,residence,property,otherplans,housing,cards,job,liable,tele,foreign'.split(',')
    target = 'Default'
    sensitive = 'age'
    options = {
        'checkingstatus1': 'A11,A12,A13,A14',
        'duration': 'continuous',
        'history': 'A30,A31,A32,A33,A34',
        'purpose': 'A40,A41,A42,A43,A44,A45,A46,A47,A48,A49,A410',
        'amount': 'continuous',
        'savings': 'A61,A62,A63,A64,A65',
        'employ': 'A71,A72,A73,A74,A75',
        'installment': 'i1,i2,i3,i4',
        'status': 'A91,A92,A93,A94',
        'others': 'A101,A102,A103',
        'residence': 'r1,r2,r3,r4',
        'property': 'A121,A122,A123,A124',
        'age': 'continuous',
        'otherplans': 'A141,A142,A143',
        'housing': 'A151,A152,A153',
        'cards': 'c1,c2,c3,c4',
        'job': 'A171,A172,A173,A174',
        'liable': 'l1,l2',
        'tele': 'A191,A192',
        'foreign': 'A201,A202',
        'Default': 'good,bad',
    }

    buckets = {'age': [30]}
    options = {k: [s.strip() for s in sorted(options[k].split(','))] for k in options}
    fns = {
        'checkingstatus1': lambda x: onehot(x, options['checkingstatus1']),
        'duration': lambda x: continuous(x),
        'history': lambda x: onehot(x, options['history']),
        'purpose': lambda x: onehot(x, options['purpose']),
        'amount': lambda x: continuous(x),
        'savings': lambda x: onehot(x, options['savings']),
        'employ': lambda x: onehot(x, options['employ']),
        'installment': lambda x: onehot(x, options['installment']),
        'status': lambda x: onehot(x, options['status']),
        'others': lambda x: onehot(x, options['others']),
        'residence': lambda x: onehot(x, options['residence']),
        'property': lambda x: onehot(x, options['property']),
        'age': lambda x: bucket(x, buckets['age']),
        'otherplans': lambda x: onehot(x, options['otherplans']),
        'housing': lambda x: onehot(x, options['housing']),
        'cards': lambda x: onehot(x, options['cards']),
        'job': lambda x: onehot(x, options['job']),
        'liable': lambda x: onehot(x, options['liable']),
        'tele': lambda x: onehot(x, options['tele']),
        'foreign': lambda x: onehot(x, options['foreign']),
        'Default': lambda x: onehot(x.strip('.'), options['Default']),
    }

    D = {}
    for dataset, phase in [(train_set, 'training'), (test_set, 'test')]:
        data = dataset
        X = []
        Y = []
        A = []
        print(phase)
        for r in data:
            row = [s for s in r]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([''], ['|1x3 Cross validator']):
                continue
            newrow, label, sens_att = parse_row(row, headers, headers_use)
            X.append(newrow)
            Y.append(label)
            A.append(sens_att)

        npX = np.array(X)
        npY = np.array(Y)
        npA = np.array(A)
        D[phase] = {}
        D[phase]['X'] = npX
        D[phase]['Y'] = npY
        D[phase]['A'] = npA

    mn = np.mean(D['training']['X'], axis=0)
    std = np.std(D['training']['X'], axis=0)
    D['training']['X'] = whiten(D['training']['X'], mn, std)
    D['test']['X'] = whiten(D['test']['X'], mn, std)

    f = open(hd_file, 'w')
    i = 0
    for h in headers_use:
        if options[h] == 'continuous':
            f.write('{:d},{}\n'.format(i, h))
            i += 1
        elif options[h][0] == 'buckets':
            for b in buckets[h]:
                colname = '{}_{:d}'.format(h, b)
                f.write('{:d},{}\n'.format(i, colname))
                i += 1
        else:
            for opt in options[h]:
                colname = '{}_{}'.format(h, opt)
                f.write('{:d},{}\n'.format(i, colname))
                i += 1

    n = D['training']['X'].shape[0]
    shuf = np.random.permutation(n)
    valid_pct = 0.3
    valid_ct = int(n * valid_pct)
    valid_inds = shuf[:valid_ct]
    train_inds = shuf[valid_ct:]

    np.savez(f_out_np, x_train=D['training']['X'], x_test=D['test']['X'],
             y_train=D['training']['Y'], y_test=D['test']['Y'],
             attr_train=D['training']['A'], attr_test=D['test']['A'],
             train_inds=train_inds, valid_inds=valid_inds)
    print('process finished...')
