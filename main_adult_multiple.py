import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AdultDataset_MULTIPLE as AdultDataset
from models import MLPNet, RegNet
from utils import conditional_errors
import coral

# result lists whose sensitive attribute is race.
race_overall_acc_list = []  # acc
race_dp_gap_list = []  # dp
race_equalized_odds_y0_list = []  # eo
race_equalized_odds_mean_list = []  # eodds
# result lists whose sensitive attribute is sex.
sex_overall_acc_list = []  # acc
sex_dp_gap_list = []  # dp
sex_equalized_odds_y0_list = []  # eo
sex_equalized_odds_mean_list = []  # eodds

# 1.compared models
# model = "mlp"
model = "caf"

# 2.batchsize
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
adult_train = AdultDataset(root_dir='./datasets', phase='train', tar_attr="income")
adult_test = AdultDataset(root_dir='./datasets', phase='test', tar_attr="income")
train_loader = DataLoader(adult_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(adult_test, batch_size=BATCH_SIZE, shuffle=False)

# train data info
input_dim = adult_train.xdim
num_classes = 2
num_groups = 2
train_target_attrs_sex = np.argmax(adult_train.A_sex, axis=1)
train_target_attrs_race = np.argmax(adult_train.A_race, axis=1)
train_target_labels = np.argmax(adult_train.Y, axis=1)
train_idx_sex = train_target_attrs_sex == 0
train_idx_race = train_target_attrs_race == 0

# test data info
target_insts = torch.from_numpy(adult_test.X).float().to(device)
target_labels = np.argmax(adult_test.Y, axis=1)
target_attrs_sex = np.argmax(adult_test.A_sex, axis=1)
target_attrs_race = np.argmax(adult_test.A_race, axis=1)
test_idx_sex = target_attrs_sex == 0
test_idx_race = target_attrs_race == 0
conditional_idx = target_labels == 0

# conditional probability Pr(A = 0 & Y = 1| Y = 1)
attr_sex_label = np.mean(np.logical_and(test_idx_sex, ~conditional_idx)) / np.mean(~conditional_idx)
attr_race_label = np.mean(np.logical_and(test_idx_race, ~conditional_idx)) / np.mean(~conditional_idx)
# classification error
race_cls_error, race_error_0, race_error_1 = 0.0, 0.0, 0.0
sex_cls_error, sex_error_0, sex_error_1 = 0.0, 0.0, 0.0
# DP
race_pred_0, race_pred_1 = 0.0, 0.0
sex_pred_0, sex_pred_1 = 0.0, 0.0
# Eodds
race_cond_00, race_cond_01, race_cond_10, race_cond_11 = 0.0, 0.0, 0.0, 0.0
sex_cond_00, sex_cond_01, sex_cond_10, sex_cond_11 = 0.0, 0.0, 0.0, 0.0


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
        for xs, ys, attrs_sex, attrs_race in train_loader:
            xs, ys, attrs_sex, attrs_race = xs.to(device), ys.to(device), attrs_sex.to(device), attrs_race.to(device)
            optimizer.zero_grad()
            ypreds = net(xs)
            loss = F.nll_loss(ypreds, ys)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
    # Test.
    net.eval()
    preds_labels = torch.max(net(target_insts), 1)[1].cpu().numpy()
    race_cls_error, race_error_0, race_error_1 = conditional_errors(preds_labels, target_labels, target_attrs_race)
    sex_cls_error, sex_error_0, sex_error_1 = conditional_errors(preds_labels, target_labels, target_attrs_sex)
    race_pred_0, race_pred_1 = np.mean(preds_labels[test_idx_race]), np.mean(preds_labels[~test_idx_race])
    sex_pred_0, sex_pred_1 = np.mean(preds_labels[test_idx_sex]), np.mean(preds_labels[~test_idx_sex])
    race_cond_00 = np.mean(preds_labels[np.logical_and(test_idx_race, conditional_idx)])
    race_cond_10 = np.mean(preds_labels[np.logical_and(~test_idx_race, conditional_idx)])
    race_cond_01 = np.mean(preds_labels[np.logical_and(test_idx_race, ~conditional_idx)])
    race_cond_11 = np.mean(preds_labels[np.logical_and(~test_idx_race, ~conditional_idx)])
    sex_cond_00 = np.mean(preds_labels[np.logical_and(test_idx_sex, conditional_idx)])
    sex_cond_10 = np.mean(preds_labels[np.logical_and(~test_idx_sex, conditional_idx)])
    sex_cond_01 = np.mean(preds_labels[np.logical_and(test_idx_sex, ~conditional_idx)])
    sex_cond_11 = np.mean(preds_labels[np.logical_and(~test_idx_sex, ~conditional_idx)])
    # save data
    race_overall_acc_list.append(1.0 - race_cls_error)
    race_dp_gap_list.append(np.abs(race_pred_0 - race_pred_1))
    race_equalized_odds_y0_list.append(np.abs(race_cond_00 - race_cond_10))
    race_equalized_odds_mean_list.append(
        (np.abs(race_cond_00 - race_cond_10) + np.abs(race_cond_01 - race_cond_11)) * 0.5)
    sex_overall_acc_list.append(1.0 - sex_cls_error)
    sex_dp_gap_list.append(np.abs(sex_pred_0 - sex_pred_1))
    sex_equalized_odds_y0_list.append(np.abs(sex_cond_00 - sex_cond_10))
    sex_equalized_odds_mean_list.append(
        (np.abs(sex_cond_00 - sex_cond_10) + np.abs(sex_cond_01 - sex_cond_11)) * 0.5)
    # save to the file
    race_data = np.column_stack(
        (race_overall_acc_list, race_dp_gap_list, race_equalized_odds_y0_list, race_equalized_odds_mean_list))
    np.savetxt('./adult_multiple_points/mlp_race.txt', race_data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
    sex_data = np.column_stack(
        (sex_overall_acc_list, sex_dp_gap_list, sex_equalized_odds_y0_list, sex_equalized_odds_mean_list))
    np.savetxt('./adult_multiple_points/mlp_sex.txt', sex_data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
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
            for xs, ys, attrs_sex, attrs_race in train_loader:
                xs, ys, attrs_race, attrs_sex = xs.to(device), ys.to(device), attrs_race.to(device), attrs_sex.to(device)
                optimizer.zero_grad()
                ypreds, ypreds_1 = net(xs)
                ypreds_y0 = ypreds_1[ys == 0]
                ypreds_y1 = ypreds_1[ys == 1]
                loss = F.nll_loss(ypreds, ys)
                running_loss += loss.item()
                # calculate fairness loss
                race_a0 = attrs_race == 0
                race_a1 = attrs_race == 1
                sex_a0 = attrs_sex == 0
                sex_a1 = attrs_sex == 1
                race_a0_hat = ypreds[race_a0]
                race_a1_hat = ypreds[race_a1]
                sex_a0_hat = ypreds[sex_a0]
                sex_a1_hat = ypreds[sex_a1]
                reg_race = coral.CORAL(race_a0_hat, race_a1_hat)
                reg_sex = coral.CORAL(sex_a0_hat, sex_a1_hat)
                # entire loss
                loss += reg_lmbda * (reg_race + reg_sex) * 0.5
                corr_sum += reg_race
                corr_sum += reg_sex
                loss.backward()
                optimizer.step()
        # Test.
        net.eval()
        preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
        race_cls_error, race_error_0, race_error_1 = conditional_errors(preds_labels, target_labels,
                                                                        target_attrs_race)
        sex_cls_error, sex_error_0, sex_error_1 = conditional_errors(preds_labels, target_labels,
                                                                     target_attrs_sex)
        race_pred_0, race_pred_1 = np.mean(preds_labels[test_idx_race]), np.mean(preds_labels[~test_idx_race])
        sex_pred_0, sex_pred_1 = np.mean(preds_labels[test_idx_sex]), np.mean(preds_labels[~test_idx_sex])
        race_cond_00 = np.mean(preds_labels[np.logical_and(test_idx_race, conditional_idx)])
        race_cond_10 = np.mean(preds_labels[np.logical_and(~test_idx_race, conditional_idx)])
        race_cond_01 = np.mean(preds_labels[np.logical_and(test_idx_race, ~conditional_idx)])
        race_cond_11 = np.mean(preds_labels[np.logical_and(~test_idx_race, ~conditional_idx)])
        sex_cond_00 = np.mean(preds_labels[np.logical_and(test_idx_sex, conditional_idx)])
        sex_cond_10 = np.mean(preds_labels[np.logical_and(~test_idx_sex, conditional_idx)])
        sex_cond_01 = np.mean(preds_labels[np.logical_and(test_idx_sex, ~conditional_idx)])
        sex_cond_11 = np.mean(preds_labels[np.logical_and(~test_idx_sex, ~conditional_idx)])
        # save data
        race_overall_acc_list.append(1.0 - race_cls_error)
        race_dp_gap_list.append(np.abs(race_pred_0 - race_pred_1))
        race_equalized_odds_y0_list.append(np.abs(race_cond_00 - race_cond_10))
        race_equalized_odds_mean_list.append(
            (np.abs(race_cond_00 - race_cond_10) + np.abs(race_cond_01 - race_cond_11)) * 0.5)
        sex_overall_acc_list.append(1.0 - sex_cls_error)
        sex_dp_gap_list.append(np.abs(sex_pred_0 - sex_pred_1))
        sex_equalized_odds_y0_list.append(np.abs(sex_cond_00 - sex_cond_10))
        sex_equalized_odds_mean_list.append(
            (np.abs(sex_cond_00 - sex_cond_10) + np.abs(sex_cond_01 - sex_cond_11)) * 0.5)
        print("reg lambda=", reg_lmbda)
    # save to the file
    race_data = np.column_stack((race_overall_acc_list, race_dp_gap_list, race_equalized_odds_y0_list, race_equalized_odds_mean_list))
    np.savetxt('./adult_multiple_points/caf_race.txt', race_data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
    sex_data = np.column_stack(
        (sex_overall_acc_list, sex_dp_gap_list, sex_equalized_odds_y0_list, sex_equalized_odds_mean_list))
    np.savetxt('./adult_multiple_points/caf_sex.txt', sex_data, header="acc, DP Gap, EO Gap, EOdds Gap", fmt='%.4f')
else:
    raise NotImplementedError("{} not supported.".format(model))
