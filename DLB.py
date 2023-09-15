import os
import numpy as np
import cv2 as cv
import torch
import torchvision
from matplotlib import pyplot as plt
import argparse
import configparser
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset

def main():

    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD'], help='Select descriptor')
    parser.add_argument('--dataset', type=str, default='GardensPoint', choices=['GardensPoint', 'StLucia', 'SFU'], help='Select dataset')
    args = parser.parse_args()

    # plt.ion()

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'StLucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'SFU':
        dataset = SFUDataset()
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    print(imgs_db[0].shape)

    diff_matrix = get_diff_matrix(imgs_db, imgs_q)
    diff_matrix = matrix_contranst_enhancement(diff_matrix, 10)
    template_best_seq_sum = matrix_matching(diff_matrix, (0.8,1.2,0.1,10))

    diff_matrix = get_best_trajectory(template_best_seq_sum, diff_matrix, 10, 10)


def get_diff_matrix(images_database, images_query):

    diff_matrix = np.zeros(images_database.shape[0], images_query.shape[0])
    for i in range(images_query.shape[0]):
        query_sequence = np.hstack(images_query[i]*images_query.shape[0])
        diff_matrix[:,i] = np.sum(abs(query_sequence - images_database), axis = 0)/images_database.shape[1]

    return diff_matrix

def matrix_contranst_enhancement(difference_matrix, r_window):
    diff_matrix = difference_matrix
    for i in range(diff_matrix.shape[1]):
        for j in range(0,diff_matrix.shape[0], r_window/2):
            a = max(0, j-r_window/2)
            b = min(diff_matrix.shape[0], j+r_window/2)
            d_window = difference_matrix[j,a:b]
            mean = np.mean(d_window)
            sigma = np.std(d_window)

            diff_matrix[j,a:b] = (d_window-mean)/sigma


    return diff_matrix

def matrix_matching(difference_matrix, match_param ):
    v_max, v_min, v_step, ds = match_param[0], match_param[1], match_param[2], match_param[3]
    template_best_seq_sum = []

    for i in range(difference_matrix.shape[0]):
        best_sequence_sum = 10000
        best_sequence = (0,0)

        for grad in range(v_min, v_max, v_step):
            x = np.arange(difference_matrix.shape[1] - ds, difference_matrix.shape[1], 1)
            y = (grad*x + i).astype(int)
            
            sequence_sum = np.sum(difference_matrix(x,y))
            if sequence_sum < best_sequence_sum:
                best_sequence_sum = sequence_sum

            best_sequence = (x,y)


        template_best_seq_sum.append(best_sequence)

def get_best_trajectory(best_templates, diff_matrix, r_window, mu):
    best_template_scores = []
    min_trajectory = 1000
    for i in range(len(best_templates)):
        (x,y) = best_templates[i]
        best_template_scores.append(np,sum(diff_matrix[a,b]))

    for i in range(len(best_templates)):
        min_trajectory = min(best_template_scores[i:i+r_window])
        # min_trajectory = min(all(traj_score-min_trajectory > mu for traj_score in best_template_scores) The trajectory uniqueness parameter needs to be thought through.

    min_idx = best_template_scores.index(min_trajectory)

    return diff_matrix[best_templates[min_idx]]



if __name__ == "__main__":
    main()