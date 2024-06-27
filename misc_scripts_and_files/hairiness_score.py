'''
Name: Luning Ding
Date: July 5, 2023
'''

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from classes import HairnessRatingDataset, ResNet, Bottleneck, BasicBlock
from functions import show_landmarks_batch, train_hairiness_model, crop, hairiness_rating
from collections import defaultdict

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def hairiness_score_main():
    root = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    whole_bee_dict = defaultdict(list)
    rating_path = os.path.join(root, 'Hairiness Manual Score_4.csv')
    rating_frame = pd.read_csv(rating_path)
    rank_dict = rating_frame.set_index(['7'])['1.8'].to_dict()

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0],
                             std=[1, 1, 1])
    ])

    rating_dataset = HairnessRatingDataset(
        csv_file = rating_path,
        root_dir = root,
        transform=data_transform)

    dataset_loader = DataLoader(rating_dataset, batch_size=8,
                                shuffle=True, num_workers=2)

    train_len = int(len(rating_dataset) * 0.8)
    val_len = len(rating_dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(rating_dataset, [train_len, val_len])
    print('training set length:', len(train_set))
    print('validataion set length:', len(val_set))

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for inputs in dataset_loader:
        psum += inputs['image'].sum(axis=[0, 2, 3])
        psum_sq += (inputs['image'] ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(rating_dataset) * 256 * 256

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])

    resnet18_config = ResNetConfig(block=BasicBlock,
                                   n_blocks=[2, 2, 2, 2],
                                   channels=[64, 128, 256, 512])

    resnet152_config = ResNetConfig(block=Bottleneck,
                                    n_blocks=[3, 8, 36, 3],
                                    channels=[64, 128, 256, 512])
    OUTPUT_DIM = 20
    resnet50 = ResNet(resnet50_config, OUTPUT_DIM)
    resnet18 = ResNet(resnet18_config, OUTPUT_DIM)
    resnet152 = ResNet(resnet152_config, OUTPUT_DIM)

    resnet = ResNet(resnet50_config, OUTPUT_DIM).to(device)
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    idx_tensor = [idx for idx in range(20)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
    rank_list = []
    for i in idx_tensor:
        rank_list.append(rank_dict[i.item()])

    model_path = root + '/models/state_dict_model_train_22bins_new.pt'
    model = torch.load(model_path)
    model.eval()

    plt.scatter(whole_bee_dict['predicted_score'], whole_bee_dict['Entropy_mean'])
    plt.xlabel('predicted_score')
    plt.ylabel('Entropy_mean')

    plt.show()

    score_df = pd.DataFrame(whole_bee_dict['predicted_score'], columns=['predicted_score'])
    entropy_df = pd.DataFrame(whole_bee_dict['Entropy_mean'], columns=['Entropy_mean'])

    print('Correlation between ground_truth_rating and Entropy_mean: ',
          score_df['predicted_score'].corr(entropy_df['Entropy_mean']))

    artificial_bees_dir = root + '/artificial_bees'
    hairiness_rating(artificial_bees_dir, model, device, rank_dict, data_transform)

    plt.scatter(whole_bee_dict['predicted_score'], whole_bee_dict['Entropy_mean'])
    plt.xlabel('predicted_score')
    plt.ylabel('Entropy_mean')

    plt.show()

    score_df = pd.DataFrame(whole_bee_dict['predicted_score'], columns=['predicted_score'])
    entropy_df = pd.DataFrame(whole_bee_dict['Entropy_mean'], columns=['Entropy_mean'])

    print('Correlation between ground_truth_rating and Entropy_mean: ',
          score_df['predicted_score'].corr(entropy_df['Entropy_mean']))
