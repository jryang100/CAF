import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LAW_SEX
from models import MLPNet, RegNet
from utils import conditional_errors
import coral

overall_acc_list = []  # acc
dp_gap_list = []  # dp
equalized_odds_y0_list = []  # eo
equalized_odds_mean_list = []  # eodds

# 1.compared models
model = "mlp"
# model = "caf"

# 2.batch size
BATCH_SIZE = 512

# 3.learning rate lr
learning_rate = 1.0

# Compile and configure all the model parameters.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)
# Set random number seed.
np.random.seed(42)
torch.manual_seed(42)
dtype = np.float32

lawdata = pd.read_csv("./datasets/law.csv").values
num_classes = 2
num_groups = 2
num_insts = lawdata.shape[0]
indices = np.arange(num_insts)
np.random.shuffle(indices)
lawdata = lawdata[indices]

# Partition the dataset into train and test split.
ratio = 0.7
num_train = int(num_insts * ratio)
law_train = LAW_SEX(lawdata[:num_train, :])
law_test = LAW_SEX(lawdata[num_train:, :])
train_loader = DataLoader(law_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(law_test, batch_size=BATCH_SIZE, shuffle=False)

# train data info
input_dim = law_train.xdim
idx = law_train.attrs == 0

# test data info
target_insts = torch.from_numpy(law_test.insts).float().to(device)
target_labels = law_test.labels
target_attrs = law_test.attrs
test_idx = target_attrs == 0
conditional_idx = target_labels == 0

cls_error, error_0, error_1 = 0.0, 0.0, 0.0
pred_0, pred_1 = 0.0, 0.0
cond_00, cond_01, cond_10, cond_11 = 0.0, 0.0, 0.0, 0.0

configs = {"num_classes": num_classes, "num_groups": num_groups, "num_epochs": 50, "batch_size": BATCH_SIZE,
           "lr": learning_rate, "input_dim": input_dim, "hidden_layers": [30], "adversary_layers": [40]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]
lr = configs["lr"]

if model == "mlp":
    print("model name: MLP")
    print("Hyperparameter setting = {}.".format(configs))
    net = MLPNet(configs).to(device)
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    net.train()
    for t in range(num_epochs):
        running_loss = 0.0
        for xs, ys, attrs in train_loader:
            xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
            optimizer.zero_grad()
            ypreds = net(xs)
            loss = F.nll_loss(ypreds, ys)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
    # Test.
    net.eval()
    preds_labels = torch.max(net(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
    # save
    overall_acc_list.append(1.0 - cls_error)
    dp_gap_list.append(np.abs(pred_0 - pred_1))
    equalized_odds_y0_list.append(np.abs(cond_00 - cond_10))
    equalized_odds_mean_list.append((np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)) * 0.5)
    data = np.column_stack((overall_acc_list, dp_gap_list, equalized_odds_y0_list, equalized_odds_mean_list))
    np.savetxt('./law_sex_points/mlp.txt', data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
elif model == "caf":
    print("model name: CAF")
    print("Hyperparameter setting = {}.".format(configs))
    for reg_lmbda in [i * 0.01 for i in range(1, 1001, 10)]:
        net = RegNet(configs).to(device)
        optimizer = optim.Adadelta(net.parameters(), lr=lr)
        net.train()
        for t in range(num_epochs):
            running_loss = 0.0
            corr_sum = 0.0
            for xs, ys, attrs in train_loader:
                xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
                optimizer.zero_grad()
                ypreds, ypreds_1 = net(xs)
                ypreds_y0 = ypreds_1[ys == 0]
                ypreds_y1 = ypreds_1[ys == 1]
                loss = F.nll_loss(ypreds, ys)
                running_loss += loss.item()
                a0 = attrs == 0
                a1 = attrs == 1
                a0_hat = ypreds[a0]
                a1_hat = ypreds[a1]
                reg = coral.CORAL(a0_hat, a1_hat)
                loss += reg_lmbda * reg
                corr_sum += reg
                loss.backward()
                optimizer.step()
        # Test.
        net.eval()
        preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
        cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
        pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
        cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
        cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
        cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
        cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
        # save
        overall_acc_list.append(1.0 - cls_error)
        dp_gap_list.append(np.abs(pred_0 - pred_1))
        equalized_odds_y0_list.append(np.abs(cond_00 - cond_10))
        equalized_odds_mean_list.append((np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)) * 0.5)
        print("reg lambda=", reg_lmbda)
    data = np.column_stack((overall_acc_list, dp_gap_list, equalized_odds_y0_list, equalized_odds_mean_list))
    np.savetxt('./law_sex_points/caf.txt', data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
else:
    raise NotImplementedError("{} not supported.".format(model))
