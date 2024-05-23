import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from graphdot import Graph
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta, Constant
from tqdm import tqdm
import random


def filter_process_det_by_id(dataset_name, raw_det_annos, id_list, score_thresh=0.3):
    filter_det = []
    for idx in id_list:
        for det in raw_det_annos:
            if idx == det["frame_id"]:
                filter_det.append(det)
    process_det = process_det_annos(dataset_name, filter_det, score_thresh)
    return process_det


def process_det_annos(dataset_name, raw_det_annos, score_thresh):
    """

    :param dataset_name:
    :param raw_det_annos:
    :param score_thresh:
    :return: {frame_id:[{"label":ndarray(N,), "pos":ndarray(N, 2)}, ... ],
               ....
             }
    """
    process_det = {}
    if dataset_name == 'kitti':
        label_map = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    elif dataset_name == 'lyft':
        label_map = {'Car': 1, 'Truck': 2, 'Bus': 3, 'Emergency_vehicle': 4, 'Other_vehicle': 5,
                     'Motorcycle': 6, 'Bicycle': 7, 'Pedestrian': 8, 'Animal': 9}
    elif dataset_name == 'suscape':
        label_map = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Scooter': 4, 'Truck': 5,
                     'Bicycle': 6, 'Van': 7, 'Bus': 8}

    else:
        raise NotImplementedError

    raw_det_annos = list(raw_det_annos)
    for det in raw_det_annos:
        choose_idx = det["score"] >= score_thresh
        if len(choose_idx) > 100:
            choose_idx = choose_idx & np.array([True] * 10 + [False] * (len(choose_idx) - 10))
        label = np.array([label_map[n] for n in det["name"][choose_idx]])
        pos = det["boxes_lidar"][choose_idx][:, :2]
        process_det[det["frame_id"]] = [{"label": label[i], "pos": pos[i]} for i in range(len(label))]

    return process_det


def edge_between_object(obj1, obj2, threshold):
    """
    :param obj1: 物体1，包含位置
    :param obj2: 物体2，包含位置
    :param threshold: float，表示构成edge的临界值
    :return: 两个物体是否构成edge，以及对应的权重
    """
    rel_posi = obj2['pos'] - obj1['pos']
    posi_val = np.linalg.norm(rel_posi)
    edge = False
    if posi_val <= threshold:
        edge = True

    return edge, 1.0 / posi_val


def get_ego_sur_state(stype):
    """

    :param stype: "ego" or "sur"
    :return:
    """
    state_obj = dict()
    state_obj["label"] = 0
    if stype == "ego":
        state_obj["pos"] = np.array([0, 0])
    elif stype == "sur":
        state_obj["pos"] = np.array([0, -1])

    return state_obj


def graph_generation(vertex_list, edge_list):
    g = nx.Graph()
    for vertex in vertex_list:
        idx = vertex['id']
        category = vertex['label']
        g.add_node(idx, category=category)

    for edge in edge_list:
        head = edge['head']
        tail = edge['tail']
        weight = edge['weight']
        g.add_edge(head, tail, w=weight, length=weight)

    return g


def get_vertex_edge_frameinfo(frame_info, threshold):
    """

    :param frame_info: list, [{"label":ndarray(N,), "pos":ndarray(N, 2) }, ...]
    :param threshold:
    :return:
    """
    frame_info.append(get_ego_sur_state("ego"))

    vertex_list = []
    edge_list = []
    vertex_num = len(frame_info)

    for i in range(vertex_num):
        vertex = {'label': frame_info[i]['label'], 'id': i}
        vertex_list.append(vertex)
        for j in range(vertex_num - i):
            obj1_num = i
            obj2_num = vertex_num - j - 1
            if obj1_num != obj2_num:
                obj1 = frame_info[obj1_num]
                obj2 = frame_info[obj2_num]
                e, edge_weight = edge_between_object(obj1, obj2, threshold)
                if e:
                    edge_1 = {'head': obj1_num, 'tail': obj2_num, 'weight': edge_weight}
                    edge_list.append(edge_1)

                    edge_2 = {'head': obj2_num, 'tail': obj1_num, 'weight': edge_weight}
                    edge_list.append(edge_2)

    sur_state = get_ego_sur_state("sur")
    vertex = {'label': sur_state['label'], 'id': vertex_num}
    vertex_list.append(vertex)

    e, edge_weight = edge_between_object(frame_info[-1], sur_state, threshold)

    edge_1 = {'head': vertex_num - 1, 'tail': vertex_num, 'weight': edge_weight}
    edge_list.append(edge_1)

    edge_2 = {'head': vertex_num, 'tail': vertex_num - 1, 'weight': edge_weight}
    edge_list.append(edge_2)

    return vertex_list, edge_list


def get_graph_from_frame(frame_info, threshold=100):
    """

    :param frame_info: list, [{"label":ndarray(N,), "pos":ndarray(N, 2) }, ...]
    :param threshold: the distance threhold of two obj
    :return:
    """
    vertex_list, edge_list = get_vertex_edge_frameinfo(frame_info, threshold)
    g = graph_generation(vertex_list, edge_list)
    return g


def similarity_among_graphs(graphs):
    knode = TensorProduct(category=KroneckerDelta(0.5))
    kedge = TensorProduct(length=SquareExponential(1.0))
    mlgk = MarginalizedGraphKernel(knode, kedge, q=0.1)
    R = mlgk([Graph.from_networkx(g, weight='w') for g in graphs])
    d = np.diag(R) ** -0.5
    K = np.diag(d).dot(R).dot(np.diag(d))
    return K


def farthest_point_sampling(key_list, distance_metrix, sample_num, init_sample=True):
    import copy
    # map idx and key
    # ["000123", "0021123",...] -> [0, 1, 2, ...]

    # find the center of gravity as p, use the point farthest from p as the first point
    if init_sample:
        dist = np.sum(distance_metrix, axis=1)
        center_point = np.argmin(dist)
        first_point = np.argmax(distance_metrix[center_point])
        sampled_points = [first_point]
    else:
        sampled_points = random.sample(list(np.arange(0, len(key_list))), 1)
        first_point = sampled_points[0]

    unsampled_points = list(np.arange(0, len(key_list)))
    unsampled_points.remove(first_point)

    while len(sampled_points) < sample_num:
        max_dist = -1
        max_point = -1
        for i in tqdm(unsampled_points):
            min_dist = float("inf")
            for j in sampled_points:
                if distance_metrix[i][j] < min_dist:
                    min_dist = distance_metrix[i][j]
            if min_dist > max_dist:
                max_dist = min_dist
                max_point = i
        sampled_points.append(max_point)
        unsampled_points.remove(max_point)

    sampled_key_list = [key_list[i] for i in sampled_points]

    return sampled_key_list, sampled_points


if __name__ == "__main__":
    from tqdm import tqdm
    import os

    os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'

    det_annos = np.load("./kitti_det_annos.npy", allow_pickle=True)
    process_det = process_det_annos('kitti', det_annos, 0.3)
    g_list = []
    key_list = []
    for key, value in tqdm(process_det.items()):
        # print(key)
        g = get_graph_from_frame(value)
        g_list.append(g)
        key_list.append(key)
        # plt.title("frame_id:{}".format(key))
        # nx.draw(g, with_labels=True, font_weight='bold')

        # plt.savefig("./graph/{}.png".format(key))
        # plt.cla()
    # for i in tqdm(range(len(g_list))):
    #     for j in tqdm(range(i+1, len(g_list))):
    #         t = [g_list[i], g_list[j]]
    #         similarity_among_graphs(t)
    g_list = g_list[:10]
    key_list = key_list[:10]
    k = similarity_among_graphs(g_list)
    print("ok")
    # farthest_point_sampling(key_list, 1-k, 5)
    None
