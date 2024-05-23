from nuscenes.nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import box_in_image
from lyft_dataset_sdk.lyftdataset import LyftDataset

import matplotlib.pyplot as plt
import networkx as nx
lyft = LyftDataset(data_path='/public_dataset/Lyft',json_path='/public_dataset/Lyft/train_data',verbose=True)

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

# 输入box的name，判断是否属于可移动物体，并根据类别给予不同的label（需要测试）
def box_label_process(name):
    label = -1
    if 'pedestrain' in name:
        label = 1
    elif 'animal' in name:
        label = -1
    elif 'car' in name:
        label = 3
    else:
        label = 2

    return label

# def box_label_process_2(name):
#     label =-1
#     mapping = {
#     'bicycle': 1,
#     'bus': 2,
#     'car': 3,
#     'emergency_vehicle': 4,
#     'motorcycle': 5,
#     'other_vehicle': 6,
#     'pedestrian': 7,
#     'truck': 8,
#     'animal': 9
#     }
#     return mapping[name]

# 得到图像中的物体的box，返回一个object_list，包括label，中心位置，速度和box四个信息
def get_box_in_image(sample_token):
    # print('here!:',sample_token)
    def range_judge(center):
        x = center[0]
        y = center[1]

        # if x < -50 or x > 50:
        #     return False
        # elif y < 0 or y > 100:
        #     return False
        # return True
        if x < -100 or x > 0:
            return False
        elif y < -50 or y > 50:
            return False
        return True

    sample = lyft.get('sample', sample_token)
    annotation_list = sample['anns']
    cm_token = sample['data']['CAM_FRONT']
    ps_token = sample['data']['LIDAR_TOP']
    cam = lyft.get('sample_data', cm_token)
    pointsensor = lyft.get('sample_data', ps_token)

    pointsensor_pose_record = lyft.get('ego_pose', pointsensor['ego_pose_token'])
    pointsensor_cs_record = lyft.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])

    cam_cs_record = lyft.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_pose_record = lyft.get('ego_pose', cam['ego_pose_token'])

    cam_intrinsic = np.array(cam_cs_record['camera_intrinsic'])
    imsize = (cam['width'], cam['height'])

    object_list = []
    for anno in annotation_list:
        box_ps = lyft.get_box(anno)
        box = box_ps.copy()

        # 将box转换到Lidar坐标系

        box_ps.translate(-np.array(pointsensor_pose_record['translation']))
        box_ps.rotate(Quaternion(pointsensor_pose_record['rotation']).inverse)

        box_ps.translate(-np.array(pointsensor_cs_record['translation']))
        box_ps.rotate(Quaternion(pointsensor_cs_record['rotation']).inverse)

        # 将box转换到camera坐标系
        box.translate(-np.array(cam_pose_record['translation']))
        box.rotate(Quaternion(cam_pose_record['rotation']).inverse)

        box.translate(-np.array(cam_cs_record['translation']))
        box.rotate(Quaternion(cam_cs_record['rotation']).inverse)

        # 如果box在camera视野中，则返回在Lidar坐标系中的box
        if box_in_image(box, cam_intrinsic, imsize):
            label = box_label_process(box_ps.name)
            velo = lyft.box_velocity(anno)
            velo[np.isnan(velo)] = 10000
            center = box_ps.center
            #             print(velo)
            if (label != -1) and (velo[0] != 10000):
                if range_judge(center):
                    # if label != -1:
                    obj = {}
                    obj['label'] = label
                    obj['center'] = center[:2]
                    # print(box_origin.center)
                    # print(box.center)
                    obj['velo'] = velo[:2]
                    # obj['box'] = box_ps
                    object_list.append(obj)

    return object_list


def get_sample_position(sample):
    lidar_data = lyft.get('sample_data', sample['data']['LIDAR_TOP'])
    pointsensor_pose_record = lyft.get('ego_pose', lidar_data['ego_pose_token'])
    return np.array(pointsensor_pose_record['translation'])

def get_velocity(sample_token):
    current_sample = lyft.get('sample', sample_token)
    has_prev = current_sample['prev'] != ''
    has_next = current_sample['next'] != ''

    if has_prev:
        first = lyft.get('sample', current_sample['prev'])
    else:
        first = current_sample

    if has_next:
        last = lyft.get('sample', current_sample['next'])
    else:
        last = current_sample

    pos_last = get_sample_position(last)
    pos_first = get_sample_position(first)
    pos_diff = pos_last - pos_first

    time_last = 1e-6 * last['timestamp']
    time_first = 1e-6 * first['timestamp']
    time_diff = time_last - time_first

    velo = pos_diff / time_diff

    return velo[:2]

def get_ego_state(sample_token):
    ego_state = {}
    ego_state['label'] = 0
    ego_state['center'] = np.array([0, 0])
    ego_state['velo'] = get_velocity(sample_token)
    return ego_state

def get_surrogate_state(sample_token):
    ego_state = {}
    ego_state['label'] = 0
    ego_state['center'] = np.array([0, -1])
    #ego_state['center'] = np.array([0, -100])
    ego_state['velo'] = get_velocity(sample_token)
    return ego_state

def get_sceneinfo(sample_token):
    object_list = get_box_in_image(sample_token)
#     print(object_list)
    ego_state = get_ego_state(sample_token)
    object_list.append(ego_state)
    return object_list

def edge_between_object(obj1, obj2, threshold):
    '''
    :param obj1: 物体1，包含位置
    :param obj2: 物体2，包含位置
    :param threshold: float，表示构成edge的临界值
    :return: 两个物体是否构成edge，以及对应的权重
    '''
    rel_posi = obj2['center'] - obj1['center']#物体1指向物体2的向量
    # 计算两个物体间的相对速度和相对距离的绝对值
    posi_val = np.linalg.norm(rel_posi)

    edge = False

    if posi_val <= threshold:
        edge = True

    return edge, 1.0 / posi_val


def get_obj_num(sample_tokens):
    total = 0
    for token in sample_tokens:
        obj_list = get_box_in_image(token)
        obj_num = len(obj_list)
        total += obj_num
    return total / len(sample_tokens)
        
    
def weight_bewteen_object(obj1, obj2):
    '''

    :param object_pair: 一对物体的特征。
    :return: 输出为物体之间相关性大小,即edge的权重,以及edge的类型
    '''
    rel_posi = obj2['center'] - obj1['center']#物体1指向物体2的向量
    rel_velo = obj2['velo'] - obj1['velo']#物体2在物体1为参考系情况下的速度

    # 计算两个物体间的相对速度和相对距离的绝对值
    posi_val = (rel_posi[0] ** 2 + rel_posi[1] ** 2) ** 0.5
    # velo_val = (rel_velo[0] ** 2 + rel_velo[1] ** 2) ** 0.5

    radial_velo = np.dot(rel_posi, rel_velo) / posi_val

    #
    edge_weight = 1.0 / posi_val * np.abs(radial_velo)


    # direction = 0

    if radial_velo >= 0:
        direction = 1
    else:
        direction = -1

    return edge_weight, direction

def sceneinfo_to_graph(sample_token, threshold):
    '''
    :param object_list:
    :return: 输出为两个vertex list和edge list。vertex表示不同类别的物体， edge表示物体之间的相关性。
    '''
    object_list = get_sceneinfo(sample_token)
    sur_state = get_surrogate_state(sample_token)
    vertex_list = []
    edge_list = []
    vertex_num = len(object_list)


    for i in range(vertex_num):
        vertex = {}
        vertex['label'] = object_list[i]['label']
        vertex['id'] = i
        vertex_list.append(vertex)
        for j in range(vertex_num-i):
            obj1_num = i
            obj2_num = vertex_num-j-1
            if obj1_num != obj2_num:
                obj1 = object_list[obj1_num]
                obj2 = object_list[obj2_num]
                e, edge_weight = edge_between_object(obj1, obj2, threshold)
                if e:
                    edge_1 = {}
                    edge_1['head'] = obj1_num
                    edge_1['tail'] = obj2_num
                    edge_1['weight'] = edge_weight
                    edge_list.append(edge_1)

                    edge_2 = {}
                    edge_2['head'] = obj2_num
                    edge_2['tail'] = obj1_num
                    edge_2['weight'] = edge_weight
                    edge_list.append(edge_2)

    vertex = {}
    vertex['label'] = sur_state['label']
    vertex['id'] = vertex_num
    vertex_list.append(vertex)

    e, edge_weight = edge_between_object(object_list[-1], sur_state, threshold)

    edge_1 = {}
    edge_1['head'] = vertex_num-1
    edge_1['tail'] = vertex_num
    edge_1['weight'] = edge_weight
    edge_list.append(edge_1)

    edge_2 = {}
    edge_2['head'] = vertex_num
    edge_2['tail'] = vertex_num-1
    edge_2['weight'] = edge_weight
    edge_list.append(edge_2)

    return vertex_list, edge_list

def dis_length_between_obj(obj1, obj2):
    rel_posi = obj2['center'] - obj1['center']#物体1指向物体2的向量
    # 计算两个物体间的相对速度和相对距离的绝对值
    posi_val = np.linalg.norm(rel_posi)

    return posi_val, 1.0 / posi_val


def print_scene_info(sample_token):
    '''
    :param object_list:
    :return: 输出为两个vertex list和edge list。vertex表示不同类别的物体， edge表示物体之间的相关性。
    '''
    object_list = get_sceneinfo(sample_token)
#     sur_state = get_surrogate_state(sample_token)
    vertex_list = []
    edge_list = []
    vertex_num = len(object_list)
    f = open(sample_token+'.txt','a')
    f.write('\n'+'*'*50)
    print('*'*50)
    for i, obj in enumerate(object_list):
        f.write('\n'+'-'*30)
        f.write('\n'+'id:'+str(i))
        f.write('\n'+'label:'+str(obj['label']))
        f.write('\n'+'center:'+str(obj['center']))
        
        
        print('-'*30)
        print('id:', i)
        print('label:', obj['label'])
        print('center:', obj['center'])

    f.write('\n'+'*'*50)
    print('*'*50)
    for i in range(vertex_num):
        vertex = {}
        vertex['label'] = object_list[i]['label']
        vertex['id'] = i
#         vertex_list.append(vertex)
        for j in range(vertex_num-i):
            obj1_num = i
            obj2_num = vertex_num-j-1
            if obj1_num != obj2_num:
                obj1 = object_list[obj1_num]
                obj2 = object_list[obj2_num]
                dis, length = dis_length_between_obj(obj1, obj2)

#                 edge_1 = {}
#                 edge_1['head'] = obj1_num
#                 edge_1['tail'] = obj2_num
#                 edge_1['weight'] = edge_weight
                f.write('\n'+'-'*30)
                f.write('\n'+'head:'+str(obj1_num))
                f.write('\n'+'tail:'+str(obj2_num))
                f.write('\n'+'distance:'+str(dis))
                f.write('\n'+'edge:'+str(length))
                
                print('-'*30)
                print('head:', obj1_num)
                print('tail:', obj2_num)
                print('distance:', dis)
                print('edge:', length)
    f.close()
                
#                 edge_list.append(edge_1)
                
if __name__ == '__main__':
    tokens =['29fd7d429450b8ac7478323ef9d5eede87fd184d74c3c36cd2e2853281043459',
             'b92e51798c633fae9351edf034178fc6e93485314e5f811bb31b0a87deff1ac6']
    for token in tokens:
        print_scene_info(token)

    vertex_list, edge_list = sceneinfo_to_graph(tokens[0], 100)
    vertex_list_2,edge_list_2 = sceneinfo_to_graph(tokens[1], 100)
    g = graph_generation(vertex_list,edge_list)
    g2 = graph_generation(vertex_list_2,edge_list_2)
    subax1 = plt.subplot(121)
    nx.draw(g, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw(g2, with_labels=True, font_weight='bold')
    plt.show()

# if __name__ == '__main__':
#     root_path = '/root/data/ss_data'
#     data_dirs = get_all_dir(root_path)
#     threshold = 100.0
#     for scene_dir in data_dirs:
#         print(scene_dir)
#         tokens = get_tokens_in_scene(scene_dir)
#         obj_num = get_obj_num(tokens)
# #         mean_s = mean_similarity(s_matrix)
#         save_path = osp.join(scene_dir, 'obj_num.npz')
#         print(obj_num)
# #         print(mean_s)
#         np.savez(save_path, num=obj_num)
    
    # nusc = NuScenes(version='v1.0-mini', dataroot='/public_dataset/nuscene', verbose=True)
    # my_scene = nusc.scene[0]
    # sample_token = my_scene['first_sample_token']
    # print(sample_token)
    # threshold = 10.0
    # vertex_list, edge_list = sceneinfo_to_graph(sample_token, threshold)
    # print(vertex_list)
    # print('*'*50)
    # print(edge_list)