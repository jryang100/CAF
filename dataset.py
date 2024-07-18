import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MEPS_RACE(Dataset):
    """
    The MEPS dataset whose sensitive attribute is race.
    """

    def __init__(self, data):
        self.insts = data[:, :-1].astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.attrs = data[:, 1].astype(np.int64)
        self.xdim = self.insts.shape[1]

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return torch.tensor(self.insts[idx]), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.attrs[idx])


class MEPS_MULTIPLE(Dataset):
    """
    The MEPS dataset with multiple sensitive attributes.
    """

    def __init__(self, data):
        self.insts = data[:, :-1].astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.attrs_race = data[:, 1].astype(np.int64)
        self.attrs_sex = data[:, 9].astype(np.int64)
        self.xdim = self.insts.shape[1]

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return torch.tensor(self.insts[idx]), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.attrs_race[idx]), \
               torch.tensor(self.attrs_sex[idx]),


class LAW_SEX(Dataset):
    """
    The LAW dataset whose sensitive attribute is sex.
    """

    def __init__(self, data):
        self.insts = data[:, :-1].astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.attrs = data[:, 8].astype(np.int64)
        self.xdim = self.insts.shape[1]

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return torch.tensor(self.insts[idx]), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.attrs[idx])


class LAW_MULTIPLE(Dataset):
    """
    The LAW dataset with multiple sensitive attributes.
    """

    def __init__(self, data):
        self.insts = data[:, :-1].astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.attrs_race = data[:, 9].astype(np.int64)
        self.attrs_sex = data[:, 8].astype(np.int64)
        self.xdim = self.insts.shape[1]

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return torch.tensor(self.insts[idx]), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.attrs_race[idx]), \
               torch.tensor(self.attrs_sex[idx])


class AdultDataset_SEX(Dataset):
    """
    The UCI Adult dataset whose sensitive attribute is sex.
    """

    def __init__(self, root_dir, phase, tar_attr, priv_attr):
        self.tar_attr = tar_attr
        self.priv_attr = priv_attr
        # load data
        self.npz_file = os.path.join(root_dir, 'adult.npz')
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1] if self.Y.shape[1] != 2 else 1
        self.adim = self.A.shape[1] if self.Y.shape[1] != 2 else 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.ydim == 1 and len(self.Y.shape) == 2:
            return torch.from_numpy(self.X[idx]).float(), \
                   self.onehot_2_int(torch.from_numpy(self.Y[idx])), \
                   self.onehot_2_int(torch.from_numpy(self.A[idx]))
        raise NotImplementedError

    def onehot_2_int(self, ts):
        if len(ts.shape) == 2:
            return torch.argmax(ts, dim=1)
        if len(ts.shape) == 1:
            return torch.argmax(ts, dim=0)
        raise NotImplementedError

    def get_A_proportions(self):
        assert len(self.A.shape) == 2
        num_class = self.A.shape[1]

        A_label = np.argmax(self.A, axis=1)
        A_proportions = []
        for cls_idx in range(num_class):
            A_proportion = np.sum(cls_idx == A_label)
            A_proportions.append(A_proportion)
        A_proportions = [a_prop * 1.0 / len(A_label) for a_prop in A_proportions]
        return A_proportions

    def get_Y_proportions(self):
        assert len(self.Y.shape) == 2
        num_class = self.Y.shape[1]

        Y_label = np.argmax(self.Y, axis=1)
        Y_proportions = []
        for cls_idx in range(num_class):
            Y_proportion = np.sum(cls_idx == Y_label)
            Y_proportions.append(Y_proportion)
        Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
        return Y_proportions

    def get_AY_proportions(self):
        assert len(self.Y.shape) == len(self.A.shape) == 2
        A_num_class = self.A.shape[1]
        Y_num_class = self.Y.shape[1]
        A_label = np.argmax(self.A, axis=1)
        Y_label = np.argmax(self.Y, axis=1)
        AY_proportions = []
        for A_cls_idx in range(A_num_class):
            Y_proportions = []
            for Y_cls_idx in range(Y_num_class):
                AY_proprtion = np.sum(np.logical_and(Y_cls_idx == Y_label, A_cls_idx == A_label))
                Y_proportions.append(AY_proprtion)
            Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
            AY_proportions.append(Y_proportions)
        return AY_proportions


class AdultDataset_MULTIPLE(Dataset):
    """
    The UCI Adult dataset with multiple sensitive attributes.
    """

    def __init__(self, root_dir, phase, tar_attr):
        self.tar_attr = tar_attr
        self.npz_file = os.path.join(root_dir, 'adult_multiple.npz')
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A_sex = self.data["attr_sex_train"]
            self.A_race = self.data["attr_race_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A_sex = self.data["attr_sex_test"]
            self.A_race = self.data["attr_race_test"]
        else:
            raise NotImplementedError

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1] if self.Y.shape[1] != 2 else 1
        self.adim_sex = self.A_sex.shape[1] if self.Y.shape[1] != 2 else 1
        self.adim_race = self.A_race.shape[1] if self.Y.shape[1] != 2 else 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.ydim == 1 and len(self.Y.shape) == 2:
            return torch.from_numpy(self.X[idx]).float(), \
                   self.onehot_2_int(torch.from_numpy(self.Y[idx])), \
                   self.onehot_2_int(torch.from_numpy(self.A_sex[idx])), \
                   self.onehot_2_int(torch.from_numpy(self.A_race[idx])),
        raise NotImplementedError

    def onehot_2_int(self, ts):
        if len(ts.shape) == 2:
            return torch.argmax(ts, dim=1)
        if len(ts.shape) == 1:
            return torch.argmax(ts, dim=0)
        raise NotImplementedError

    def get_A_sex_proportions(self):
        assert len(self.A_sex.shape) == 2
        num_class = self.A_sex.shape[1]

        A_sex_label = np.argmax(self.A_sex, axis=1)
        A_sex_proportions = []
        for cls_idx in range(num_class):
            A_sex_proportion = np.sum(cls_idx == A_sex_label)
            A_sex_proportions.append(A_sex_proportion)
        A_sex_proportions = [a_prop * 1.0 / len(A_sex_label) for a_prop in A_sex_proportions]
        return A_sex_proportions

    def get_A_race_proportions(self):
        assert len(self.A_race.shape) == 2
        num_class = self.A_race.shape[1]

        A_race_label = np.argmax(self.A_race, axis=1)
        A_race_proportions = []
        for cls_idx in range(num_class):
            A_race_proportion = np.sum(cls_idx == A_race_label)
            A_race_proportions.append(A_race_proportion)
        A_race_proportions = [a_prop * 1.0 / len(A_race_label) for a_prop in A_race_proportions]
        return A_race_proportions

    def get_Y_proportions(self):
        assert len(self.Y.shape) == 2
        num_class = self.Y.shape[1]

        Y_label = np.argmax(self.Y, axis=1)
        Y_proportions = []
        for cls_idx in range(num_class):
            Y_proportion = np.sum(cls_idx == Y_label)
            Y_proportions.append(Y_proportion)
        Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
        return Y_proportions

    def get_AY_sex_proportions(self):
        assert len(self.Y.shape) == len(self.A_sex.shape) == 2
        A_num_class = self.A_sex.shape[1]
        Y_num_class = self.Y.shape[1]
        A_label = np.argmax(self.A_sex, axis=1)
        Y_label = np.argmax(self.Y, axis=1)
        AY_proportions = []
        for A_cls_idx in range(A_num_class):
            Y_proportions = []
            for Y_cls_idx in range(Y_num_class):
                AY_proprtion = np.sum(np.logical_and(Y_cls_idx == Y_label, A_cls_idx == A_label))
                Y_proportions.append(AY_proprtion)
            Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
            AY_proportions.append(Y_proportions)
        return AY_proportions

    def get_AY_race_proportions(self):
        assert len(self.Y.shape) == len(self.A_race.shape) == 2
        A_num_class = self.A_race.shape[1]
        Y_num_class = self.Y.shape[1]
        A_label = np.argmax(self.A_race, axis=1)
        Y_label = np.argmax(self.Y, axis=1)
        AY_proportions = []
        for A_cls_idx in range(A_num_class):
            Y_proportions = []
            for Y_cls_idx in range(Y_num_class):
                AY_proprtion = np.sum(np.logical_and(Y_cls_idx == Y_label, A_cls_idx == A_label))
                Y_proportions.append(AY_proprtion)
            Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
            AY_proportions.append(Y_proportions)
        return AY_proportions


class GermanDataset(Dataset):
    """
    The GermanCredit dataset whose sensitive attribute is age.
    """

    def __init__(self, root_dir, phase, tar_attr, priv_attr):
        self.tar_attr = tar_attr
        self.priv_attr = priv_attr
        # load data
        self.npz_file = os.path.join(root_dir, 'german.npz')
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1] if self.Y.shape[1] != 2 else 1
        self.adim = self.A.shape[1] if self.Y.shape[1] != 2 else 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.ydim == 1 and len(self.Y.shape) == 2:
            return torch.from_numpy(self.X[idx]).float(), \
                   self.onehot_2_int(torch.from_numpy(self.Y[idx])), \
                   self.onehot_2_int(torch.from_numpy(self.A[idx]))
        raise NotImplementedError

    def onehot_2_int(self, ts):
        if len(ts.shape) == 2:
            return torch.argmax(ts, dim=1)
        if len(ts.shape) == 1:
            return torch.argmax(ts, dim=0)
        raise NotImplementedError

    def get_A_proportions(self):
        assert len(self.A.shape) == 2
        num_class = self.A.shape[1]

        A_label = np.argmax(self.A, axis=1)
        A_proportions = []
        for cls_idx in range(num_class):
            A_proportion = np.sum(cls_idx == A_label)
            A_proportions.append(A_proportion)
        A_proportions = [a_prop * 1.0 / len(A_label) for a_prop in A_proportions]
        return A_proportions

    def get_Y_proportions(self):
        assert len(self.Y.shape) == 2
        num_class = self.Y.shape[1]

        Y_label = np.argmax(self.Y, axis=1)
        Y_proportions = []
        for cls_idx in range(num_class):
            Y_proportion = np.sum(cls_idx == Y_label)
            Y_proportions.append(Y_proportion)
        Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
        return Y_proportions

    def get_AY_proportions(self):
        assert len(self.Y.shape) == len(self.A.shape) == 2
        A_num_class = self.A.shape[1]
        Y_num_class = self.Y.shape[1]
        A_label = np.argmax(self.A, axis=1)
        Y_label = np.argmax(self.Y, axis=1)
        AY_proportions = []
        for A_cls_idx in range(A_num_class):
            Y_proportions = []
            for Y_cls_idx in range(Y_num_class):
                AY_proprtion = np.sum(np.logical_and(Y_cls_idx == Y_label, A_cls_idx == A_label))
                Y_proportions.append(AY_proprtion)
            Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
            AY_proportions.append(Y_proportions)
        return AY_proportions
