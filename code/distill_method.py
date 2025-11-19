import sys
import os
import copy
import time
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, recall_score, cohen_kappa_score
from tqdm import tqdm
from net import ResNet2D
from fusion_code.DR_2.fusion_net_2 import IMDR
from data_glu2 import Multi_modal_data, Harvard_30k_dataset
from torch.autograd import Variable
import warnings
import matplotlib.pyplot as plt
from baseline_models import Multi_ResNet_cross, Medical_base_2DNet
warnings.filterwarnings("ignore")

def distillation_loss_logits(student_logits, teacher_logits, temperature, alpha):
    """
    Compute the distillation loss based on logits.
    :param student_logits: Output logits of the student model
    :param teacher_logits: Output logits of the teacher model
    :param temperature: Temperature parameter
    :param alpha: Balancing parameter
    """
    logits_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits / temperature, dim=1),
                                                      F.softmax(teacher_logits / temperature, dim=1)) * (alpha * temperature * temperature)
    return logits_loss

def distillation_loss_features(student_features, teacher_features, temperature, alpha):

    features_loss = nn.MSELoss()(student_features, teacher_features) * alpha
    return features_loss

def distillation_loss_proxies(student_proxies, teacher_proxies):

    proxies_loss = nn.MSELoss()(student_proxies, teacher_proxies)
    return proxies_loss

def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--model_configs", type=str, default='config.py', help="Model configuration file name.")
    parser.add_argument("--run_id", default=0, type=int, help="Run ID (default: 0)")
    parser.add_argument("--device", default="3", type=str, help="cuda:n or cpu (default: cuda:0)")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of data loader workers. (default: 0)")
    parser.add_argument("--checkpoint", default=None, type=str, help="Model checkpoint path")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--distill_epoch", default=0, type=float, help="Epoch to start distillation")
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--temperature", default=2, type=float)
    parser.add_argument("--alpha", default=2, type=float)
    parser.add_argument("--beta", default= 1, type=float)
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]'),
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument("--dataset", default="Harvard-30k DR", type=str)
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument("--condition", default="noise", type=str, help="noise/normal")
    args = parser.parse_args()

    return args

def save_results(filename, epoch, loss_meter, acc, precision, recall, f1, auc, specificity=None):
  
    with open(filename, 'a') as f:
        line = (f"Epoch: {epoch}, "
                f"Loss: {loss_meter.avg:.6f}, "
                f"Accuracy: {acc:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1 Score: {f1:.4f}, "
                f"AUC: {auc:.4f}")
        if specificity is not None:
            line += f", Specificity: {specificity:.4f}"
        f.write(line + "\n")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def val(current_epoch, val_loader, model, best_acc):
  
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())

            pred, loss = model(data, target)
            predicted = pred.argmax(dim=-1)
            loss = loss.mean()

            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            probabilities = torch.nn.functional.softmax(pred, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().numpy())

    aver_acc = correct_num / data_num

    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)

    print("All targets distribution:", np.bincount(all_targets))
    print("All predictions distribution:", np.bincount(all_predictions))

    if len(set(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')

    print(f'Validation Epoch: {current_epoch} \tLoss: {loss_meter.avg:.6f} \tAccuracy: {aver_acc:.4f}')
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')

    save_results('val_results_2c_glu_res.txt', current_epoch, loss_meter, aver_acc, precision, recall, f1, auc)

    return loss_meter.avg, best_acc


class Projector_two(nn.Module):
    def __init__(self, fc_ins=2048):
        super(MLP_two, self).__init__()
        self.linear1 = nn.Linear(fc_ins, fc_ins)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(fc_ins, fc_ins)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(fc_ins, 2048)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class Projector_fusion(nn.Module):
    def __init__(self, fc_ins=1024):
        super(MLP_fusion, self).__init__()
        self.linear1 = nn.Linear(fc_ins, fc_ins)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(fc_ins, fc_ins)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(fc_ins, 1792)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def main(args, device):
    """
    Main training loop for the student model with knowledge distillation from the teacher model.
    :param args: Arguments for the training process
    :param device: Device to use (CPU or GPU)
    """
    train_loss = 0
    correct = 0
    total = 0

    teacher_network =  IMDR(args.num_classes, args.modal_number, args.dims, args).to(device)
    student_network = ResNet2D(args.num_classes, 1, args.dims).to(device)

    checkpoint_path = '/path/to/teacher/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)
    teacher_network.load_state_dict(checkpoint["state_dict"])
    teacher_network.eval()

    student_optimizer = torch.optim.Adam(student_network.parameters(), lr=args.lr, weight_decay=1e-5)
    student_alignment_layer = Projector_two().to(device)
    teacher_alignment_layer = Projector_fusion().to(device)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(args.start_epoch, args.end_epochs + 1):
        print(f'Epoch {epoch}/{args.end_epochs}')
        student_network.train()
        losses = AverageMeter()
        best_acc = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            student_optimizer.zero_grad()

            student_logits, student_features = student_network(inputs, labels)
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_network(inputs, labels)

            aligned_student_features = student_alignment_layer(student_features)
            aligned_teacher_features = teacher_alignment_layer(teacher_features)


            logits_loss = distillation_loss_logits(student_logits, teacher_logits, args.temperature, args.alpha)
            features_loss = distillation_loss_features(aligned_student_features, aligned_teacher_features, args.temperature, args.alpha)
            proxies_loss = distillation_loss_proxies(aligned_student_features, aligned_teacher_features)

            total_loss = criterion(student_logits, labels) + logits_loss + features_loss + proxies_loss
            total_loss.backward()
            student_optimizer.step()

            losses.update(total_loss.item(), inputs[0].size(0))

        val_loss, best_acc = val(epoch, val_loader, student_network, best_acc)

if __name__ == "__main__":
    opts = parse_args()
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and opts.device != "cpu" else "cpu")
    print(f"Selected device: {device}")

    main(opts, device)
