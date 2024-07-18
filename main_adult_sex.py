import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AdultDataset_SEX
from models import MLPNet, RegNet
from utils import conditional_errors
import coral

# list used to save experimental data
overall_acc_list = []  # acc
dp_gap_list = []  # dp
equalized_odds_y0_list = []  # eo
equalized_odds_mean_list = []  # eodds

# 1.models used to compare
model = "mlp"
# model = "caf"

# 2.batch size
BATCH_SIZE = 128

# 3.learning rate lr
learning_rate = 0.01

# Compile and configure all the model parameters.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

# Set random number seed.
np.random.seed(42)
torch.manual_seed(42)
dtype = np.float32

# Load UCI Adult dataset.
adult_train = AdultDataset_SEX(root_dir='./datasets', phase='train', tar_attr="income", priv_attr="sex")
adult_test = AdultDataset_SEX(root_dir='./datasets', phase='test', tar_attr="income", priv_attr="sex")
train_loader = DataLoader(adult_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(adult_test, batch_size=BATCH_SIZE, shuffle=False)

# train data info
input_dim = adult_train.xdim
num_classes = 2
num_groups = 2
train_target_attrs = np.argmax(adult_train.A, axis=1)
train_target_labels = np.argmax(adult_train.Y, axis=1)

# test data info
target_insts = torch.from_numpy(adult_test.X).float().to(device)
target_labels = np.argmax(adult_test.Y, axis=1)
target_attrs = np.argmax(adult_test.A, axis=1)
test_idx = target_attrs == 0
conditional_idx = target_labels == 0
attr_label = np.mean(np.logical_and(test_idx, ~conditional_idx)) / np.mean(~conditional_idx)

# variable used to calculate experimental result
cls_error, error_0, error_1 = 0.0, 0.0, 0.0
pred_0, pred_1 = 0.0, 0.0
cond_00, cond_01, cond_10, cond_11 = 0.0, 0.0, 0.0, 0.0


configs = {"num_classes": num_classes, "num_groups": num_groups, "num_epochs": 50, "batch_size": BATCH_SIZE,
           "lr": learning_rate, "input_dim": input_dim, "hidden_layers": [60], "adversary_layers": [50]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]
lr = configs["lr"]

if model == "mlp":
    # Train MLPNet without debiasing.
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
    # save result
    overall_acc_list.append(1.0 - cls_error)
    dp_gap_list.append(np.abs(pred_0 - pred_1))
    equalized_odds_y0_list.append(np.abs(cond_00 - cond_10))
    equalized_odds_mean_list.append((np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)) * 0.5)
    data = np.column_stack((overall_acc_list, dp_gap_list, equalized_odds_y0_list, equalized_odds_mean_list))
    np.savetxt('./adult_sex_points/mlp.txt', data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
elif model == "caf":
    print("model name: CAF")
    print("Hyperparameter setting = {}.".format(configs))
    for reg_lmbda in [i * 0.1 for i in range(1, 1001, 10)]:
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
                # coral loss
                a0 = attrs == 0
                a1 = attrs == 1
                a0_hat = ypreds[a0]
                a1_hat = ypreds[a1]
                reg = coral.CORAL(a0_hat, a1_hat)
                # entire loss
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
        # save result
        overall_acc_list.append(1.0 - cls_error)
        dp_gap_list.append(np.abs(pred_0 - pred_1))
        equalized_odds_y0_list.append(np.abs(cond_00 - cond_10))
        equalized_odds_mean_list.append((np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)) * 0.5)
        print("reg lambda=", reg_lmbda)
    # save to the file
    data = np.column_stack((overall_acc_list, dp_gap_list, equalized_odds_y0_list, equalized_odds_mean_list))
    np.savetxt('./adult_sex_points/caf.txt', data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
else:
    raise NotImplementedError("{} not supported.".format(model))