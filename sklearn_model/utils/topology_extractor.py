# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:28:40 2021

@author: VanessaLee
"""
import cv2
import csv
import os
import numpy as np
from time import time
import networkx
import matplotlib.pyplot as plt
import glob

DIALTE_KERNEL_TYPE = cv2.MORPH_ELLIPSE  # cv2.MORPH_RECT


# 读取掩膜图片
def reade_mask_img(file_path):
    """
    :param file_path: path of mask image
    :return: a 2D numpy array
    """
    mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    # print(mask_img.shape)
    return mask_img


# 初始化链接的组件
def initial_connected_component(mask_img, ignore_thresh=2, draw=False):
    """
    extract and split the connected components in initial mask image
    :param img: the initial mask image (2D numpy array, W*H)
    :param ignore_thresh: the calcifications that less that ignore_thresh pixels are ignored
    :param draw: whether to show image during processing
    :return: an list in which every element is a connected component
    """
    image_for_draw = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # find contours
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # extract and split the connected components in mask image, ignore the very tiny calcifications
    contours_list = []
    for cid in range(len(contours)):
        if cv2.contourArea(contours[cid]) < ignore_thresh:  # ignore areas less than ignore_thresh
            continue
        c = contours[cid]  # [m, 1, 2]
        # contours_list.append(np.squeeze(c, axis=1))
        contours_list.append(c)
        if draw:
            cv2.drawContours(image_for_draw, contours, cid, (0, 255, 255), 1)

    if draw:
        cv2.imshow("cals", image_for_draw)
        cv2.waitKey(1)

    return contours_list



def dilate_component(shape, contours_list, dilate_rate, draw=False):
    """
    dilate every component in contours_list with dilate_rate
    :param shape: the shape of mask image
    :param contours_list: list of connected components
    :param dilate_rate:
    :return: a list of in which every element is a dilated connected component
    """
    image_for_draw = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)
    dilated_component_list = []
    for contour in contours_list:
        temp_img = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(temp_img, [contour], 255)  # add [] for pts to fill, otherwise only the outline would be drawn

        # dilate
        kernel = cv2.getStructuringElement(DIALTE_KERNEL_TYPE, (dilate_rate, dilate_rate))
        dilated_img = cv2.dilate(temp_img, kernel, iterations=1)
        # kernel = morphology.disk(dilate_rate)
        # dilated_img = morphology.dilation(temp_img, kernel)

        # get the board of dilated component
        new_contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(new_contours) == 1
        dilated_component_list += new_contours
        if draw:
            cv2.drawContours(image_for_draw, new_contours, 0, (0, 255, 255), 1)
      
    if draw:
        cv2.imshow("dilated", image_for_draw) 
        cv2.waitKey(1)
        #if dilate_rate in [2,4,8,16,32,64]:
            #cv2.imwrite("d_{}.png".format(dilate_rate),image_for_draw)

    return dilated_component_list


def graph_generator(shape, components, old_ad_matric):
    """
    generate a graph, every component is a vertex and the connections between components determine the edges.
    also return a networkx.Graph
    :param shape: the shape of mask image
    :param components: a list of components
    :param old_ad_matric
    :return: the adjacent matrix of the graph
    """
    num_components = len(components)
    if old_ad_matric is None:
        ad_matrix = np.zeros((num_components, num_components), dtype=np.uint8)
    else:
        ad_matrix = old_ad_matric.copy()

    kx_G = networkx.Graph()
    kx_G.add_nodes_from(list(range(num_components)))

    dilate_img = np.zeros((num_components, shape[0], shape[1]), dtype=np.uint8)
    dilate_area = np.zeros(num_components)

    for i in range(num_components):
        comp = components[i]
        cv2.fillPoly(dilate_img[i], [comp], 255)
        dilate_area[i] = len(np.where(dilate_img[i] != 0)[0])


    # check whether two components are connected
    for i in range(num_components):
        for j in range(i + 1, num_components):
            # two components that have already connected must connect with current dilation rate
            if ad_matrix[i, j] == 1:
                continue
            if np.min(components[i][:, :, 0]) > np.max(components[j][:, :, 0]) \
                    or np.min(components[j][:, :, 0]) > np.max(components[i][:, :, 0])\
                    or np.min(components[i][:, :, 1]) > np.max(components[j][:, :, 1])\
                    or np.min(components[j][:, :, 1]) > np.max(components[i][:, :, 1]):
                continue
            if dilate_area[i] + dilate_area[j] > len(np.where((dilate_img[i] + dilate_img[j]) != 0)[0]):
                ad_matrix[i, j] = ad_matrix[j, i] = 1
                try:
                    kx_G.add_edge(i, j)
                except Exception as e:
                    print('The e is:', e)
                    print("current i: ", i)
                    print("current j: ", j)

    return ad_matrix, kx_G


def graph_features_extractor(adjacency_matrix, kx_G, dilate_rate):
    """
    generate graph features according to the adjacency matrix
    :param adjacency_matrix:
    :param kx_G:
    :return: graph features
    """
    graph_features = {}
    num_vertexes = adjacency_matrix.shape[0]

    # degree matrix
    degree_matrix = np.zeros(adjacency_matrix.shape)
    for i in range(num_vertexes):
        assert np.sum(adjacency_matrix[:, i]) == np.sum(adjacency_matrix[i, :])
        degree_matrix[i, i] = np.sum(adjacency_matrix[:, i])
    graph_features["AverageVertexDegree_{}".format(dilate_rate)] = np.sum(degree_matrix) / num_vertexes
    graph_features["MaximumVertexDegree_{}".format(dilate_rate)] = np.max(degree_matrix)

    # Connected Component
    subgraphs = [kx_G.subgraph(c).copy() for c in networkx.algorithms.components.connected_components(kx_G)]
    subgraph_length = [len(c) for c in subgraphs]
    graph_features["NumberofSubgraphs_{}".format(dilate_rate)] = len(subgraph_length)
    # assert len(subgraph_vertexes) == len(subgraphs)

    # assert len(subgraph_vertexes) == num_subgraphs
    graph_features["GiantConnectedComponentRatio_{}".format(dilate_rate)] = np.max(subgraph_length) / num_vertexes
    graph_features["PercentageofIsolatedPoints_{}".format(dilate_rate)] = np.sum(np.array(subgraph_length) == 1) / num_vertexes

    # eccentricity
    kx_eccentricity = []
    for s in range(len(subgraphs)):
        ecc = networkx.algorithms.distance_measures.eccentricity(subgraphs[s])
        kx_eccentricity += [i for i in ecc.values()]
    # assert np.all(sorted(np.array(kx_eccentricity)) == sorted(eccentricity))
    graph_features["AverageVertexEccentricity_{}".format(dilate_rate)] = np.sum(kx_eccentricity) / num_vertexes
    graph_features["Diameter_{}".format(dilate_rate)] = np.max(kx_eccentricity)

    # clustering coefficient
    kx_clustering_coefficient = []
    for s in range(len(subgraphs)):
        # coe = networkx.algorithms.cluster.clustering(subgraphs[s])
        # kx_clustering_coefficient += [i for i in coe.values()]
        coe = 2 * float(len(subgraphs[s].edges)) / (len(subgraphs[s]) * len(subgraphs[s]))
        kx_clustering_coefficient += [coe]
    # assert np.all(sorted(np.array(kx_clustering_coefficient)) == sorted(clustering_coefficient))

    graph_features["AverageClusteringCoefficient_{}".format(dilate_rate)] = np.sum(kx_clustering_coefficient) / num_vertexes

    return graph_features


def feature_extractor_main(mask_path, show=False):
    """
    main function
    :param mask_path: path of mask image
    :param show: whether to show
    :return: a feature list (512, )
    """
    mask_img = reade_mask_img(mask_path)
    contours_list = initial_connected_component(mask_img)
    # assert len(contours_list) == 0
    print("number of components %d" % len(contours_list))

    all_graph_features = dict()
    old_ad_matric = None
    time_begin = time()
    for dilate_rate in range(1, 65):
        dilated_component_list = dilate_component(mask_img.shape, contours_list, dilate_rate, show)
        ad_matrix, kx_G = graph_generator(mask_img.shape, dilated_component_list, old_ad_matric)
        if dilate_rate == 2:
            networkx.draw(kx_G,pos=networkx.spring_layout(kx_G),with_labels=True)
            plt.show()
        graph_features = graph_features_extractor(ad_matrix, kx_G, dilate_rate)
        all_graph_features.update(graph_features)
        old_ad_matric = ad_matrix

        if graph_features["NumberofSubgraphs_{}".format(dilate_rate)] == 1:  # the topological features would not change any more when dilating more
            for i in range(dilate_rate+1, 65):
                all_graph_features.update(graph_features)
            break

    print("len features: ", len(all_graph_features))
    # assert len(all_graph_features) == 512 and None not in all_graph_features.values()
    print("time: %d s" % (time() - time_begin))
    return all_graph_features


#
if __name__ == '__main__':

    mask_path = r"D:\AllExploreDownloads\IDM\Topo"
    benign_features = []
    malignant_features = []

    for case_mask_img in os.listdir(mask_path):
        case_num = case_mask_img[0]
        if case_num == 'B':
            continue
        print("mask %s" % case_mask_img)
        case_img_path = os.path.join(mask_path, case_mask_img)

        topofs = feature_extractor_main(case_img_path)
        topofs["image"] = case_mask_img

        keys, values = [], []
        for key, value in topofs.items():
            keys.append(key)
            values.append(value)

        with open("./features/"+ case_mask_img.replace("bmp","csv"), "w", newline='') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(keys)
            csvwriter.writerow(values)


