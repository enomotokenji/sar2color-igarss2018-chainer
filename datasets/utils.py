import os


def read_imlist(root_dir, txt_imlist):
    with open(txt_imlist, 'r') as f:
        ret = [os.path.join(root_dir, path.strip()) for path in f.readlines()]
    return ret


def train_test_split_imlist(imlist_sar, imlist_rgnir, rate_train):
    n_train = len(imlist_sar) * rate_train
    dirname_ = ''
    for i, path in enumerate(imlist_sar):
        dirname = os.path.basename(os.path.dirname(path))
        if not dirname_ == dirname:
            if i > n_train:
                th_train = i
                break
            dirname_ = dirname

    train_sar = imlist_sar[:th_train]
    test_sar = imlist_sar[th_train:]
    train_rgnir = imlist_rgnir[:th_train]
    test_rgnir = imlist_rgnir[th_train:]

    print('Train size: {}, Test size: {}'.format(len(train_sar), len(test_sar)))

    return train_sar, test_sar, train_rgnir, test_rgnir
