import copy
import datetime
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree


def read_txt():
    startpts_path = 'c:/Users/RTX2080Ti/Desktop/西安二环内路网1/startpts.txt'
    endpts_path = 'c:/Users/RTX2080Ti/Desktop/西安二环内路网1/endpts.txt'
    nodes_path = 'c:/Users/RTX2080Ti/Desktop/西安二环内路网1/nodes.txt'
    midpts_path = 'c:/Users/RTX2080Ti/Desktop/西安二环内路网1/mid_points.txt'

    f = open(nodes_path, 'r', encoding='UTF-8')
    text = f.readlines()

    nodes = []
    for i in range(1, len(list(text))):
        nodes.append([round(float(text[i].split(",")[-2]), 6), round(float(text[i].split(",")[-1]), 6)])

    f = open(startpts_path, 'r', encoding='UTF-8')
    text1 = f.readlines()

    f = open(endpts_path, 'r', encoding='UTF-8')
    text2 = f.readlines()

    f = open(midpts_path, 'r', encoding='UTF-8')
    text3 = f.readlines()

    startpts = []
    endpts = []
    midpts = []
    edges_info = []
    for i in range(1, len(list(text1))):
        startpts.append([round(float(text1[i].split(",")[-2]), 6), round(float(text1[i].split(",")[-1]), 6)])
        endpts.append([round(float(text2[i].split(",")[-2]), 6), round(float(text2[i].split(",")[-1]), 6)])
        midpts.append([round(float(text3[i].split(",")[-2]), 6), round(float(text3[i].split(",")[-1]), 6)])
        # length,oneway
        edges_info.append(
            [round(float(text3[i].split(",")[3]), 3), text3[i].split(",")[5]])

    return nodes, startpts, midpts, endpts, edges_info


# 无权有向图的邻接表，需要考虑oneway字段。
# 初始路网邻接表每项：
# 节点作为数组索引，其邻接表每项存放：邻接节点，邻接边，邻接边的几何长度（初始边权）
def generate_adjacent_list():
    spatial_join_path = 'c:/Users/RTX2080Ti/Desktop/西安二环内路网1/空间连接1.txt'
    f = open(spatial_join_path, 'r', encoding='UTF-8')
    text = f.readlines()
    print(len(list(text)))
    print(text[0].split(","))
    node_fid = []
    edge_fid = []
    edge_length = []
    # node_fid_list:target_fid edge_fid_list:join_fid edge_info:weight
    for i in range(1, len(list(text))):
        node_fid.append(int(text[i].split(",")[2]))
        edge_fid.append(int(text[i].split(",")[3]))
        edge_length.append(round(float(text[i].split(",")[10]), 2))

    nodes_list = defaultdict(list)
    for i, e in enumerate(node_fid):
        nodes_list[e].append(e)
    nodes_list = list(nodes_list.values())
    # print("nodes_list", len(list(nodes_list)), nodes_list)

    edges_list = [[] for i in range(len(list(nodes_list)))]
    count = 0
    for i in range(len(list(nodes_list))):
        for j in range(len(list(nodes_list[i]))):
            edges_list[i].append(edge_fid[count])
            count += 1
    # print("edges_list", len(list(edges_list)), edges_list)

    init_graph = [[] for i in range(len(list(nodes_list)))]
    for edges in edges_list:
        for edge in edges:
            for i in range(len(list(edge_fid))):
                # 注意node_fid是从1开始的还是从0开始的
                if edge_fid[i] == edge and node_fid[i] != edges_list.index(edges):
                    init_graph[edges_list.index(edges)].append(
                        [node_fid[i], edge_fid[i], edge_length[i]])


    def dist(pt1, pt2):
        return 1 if np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) < 0.00001 else 0

    graph = [[] for i in range(len(list(nodes)))]
    for i in range(len(list(graph))):
        sub_graph = []
        # print(init_graph[i])
        for j in range(len(list(startpts))):
            if edges_info[j][1] == "1":
                if dist(startpts[j], nodes[i]) or dist(endpts[j], nodes[i]):
                    for k in range(len(list(init_graph[i]))):
                        if j == init_graph[i][k][1]:
                            sub_graph.append(
                                [init_graph[i][k][0], init_graph[i][k][1], init_graph[i][k][2]])
            if edges_info[j][1] == "FT":
                if dist(startpts[j], nodes[i]):
                    for k in range(len(list(init_graph[i]))):
                        if j == init_graph[i][k][1]:
                            sub_graph.append(
                                [init_graph[i][k][0], init_graph[i][k][1], init_graph[i][k][2]])
            if edges_info[j][1] == "TF":
                if dist(endpts[j], nodes[i]):
                    for k in range(len(list(init_graph[i]))):
                        if j == init_graph[i][k][1]:
                            sub_graph.append(
                                [init_graph[i][k][0], init_graph[i][k][1], init_graph[i][k][2]])
        # 输出被去掉的伪邻接节点
        # sub_list1 = edges_list[i]
        # sub_list2 = list(list(zip(*sub_graph))[1])
        # print(i, list(set(sub_list1) - set(sub_list2)))
        graph[i] = sub_graph
    # print("init_graph1", graph)
    return graph, len(list(edges_info))


def cal_angle_cos(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom


def sim_cost(pt1, pt2):
    return haversine(pt1[0], pt1[1], pt2[0], pt2[1])


# 和高德API计算结果误差很小，基本上在2cm以内。
def haversine(lon1, lat1, lon2, lat2):
    dLat = (lat2 - lat1) * np.pi / 180.0
    dLon = (lon2 - lon1) * np.pi / 180.0
    lat1 = lat1 * np.pi / 180.0
    lat2 = lat2 * np.pi / 180.0
    a = (pow(np.sin(dLat / 2), 2) + pow(np.sin(dLon / 2), 2) * np.cos(lat1) * np.cos(lat2))
    rad = 6371009
    c = 2 * np.arcsin(np.sqrt(a))
    return rad * c


def Astar(graph, nodes_pos, start_node, end_node):
    # 待判断节点集合(节点索引，对应父节点索引（起点设置为本身，用来反溯最短路径），起点到该节点最小路径距离g(x)，
    # 该节点到终点预估路径距离h(x)和起点到该节点最小路径距离g(x)两者之和是f(x)
    openlist = [[start_node, -1, 0, 0 + sim_cost(nodes_pos[start_node], nodes_pos[end_node])]]
    # 已经判断节点集合
    closelist = []
    while openlist and openlist[0][0] != end_node:
        openlist.sort(key=lambda x: x[3])
        # 查找openlist中f(x)最小的节点，记为nmin
        nmin = openlist[0]
        # 直到openlist为空或者nmin等于终点时循环结束
        # print("nmin", nmin)
        closelist.append(nmin)
        openlist.pop(0)
        adjacent_nodes = []
        # 查找Nmin的所有不属于Closelist的邻居节点
        closelist_nodes = list(list(zip(*list(closelist)))[0])
        for i in range(len(list(graph[nmin[0]]))):
            if graph[nmin[0]][i][0] not in closelist_nodes:
                adjacent_nodes.append([graph[nmin[0]][i][0], graph[nmin[0]][i][2]])
        # 如果邻接节点不为空
        if adjacent_nodes:
            # openlist节点集合
            if openlist:
                openlist_nodes = list(list(zip(*list(openlist)))[0])
                for i in range(len(list(adjacent_nodes))):
                    # 邻接节点不在openlist中，插入
                    if adjacent_nodes[i][0] not in openlist_nodes:
                        # 此时nmin[0]是这些邻接节点的父节点
                        gx = nmin[2] + adjacent_nodes[i][1]
                        fx = gx + sim_cost(nodes_pos[adjacent_nodes[i][0]], nodes_pos[end_node])
                        openlist.append([adjacent_nodes[i][0], nmin[0], gx, fx])
                    # 邻接节点在openlist中，若新的f(x)小于旧的f(x)，则更新父节点
                    if adjacent_nodes[i][0] in openlist_nodes:
                        # 对比新旧预估路径距离
                        old_fx = openlist[openlist_nodes.index(adjacent_nodes[i][0])][3]
                        gx = nmin[2] + adjacent_nodes[i][1]
                        fx = gx + sim_cost(nodes_pos[adjacent_nodes[i][0]], nodes_pos[end_node])
                        if fx < old_fx:
                            openlist[openlist_nodes.index(adjacent_nodes[i][0])][1] = nmin[0]
                            openlist[openlist_nodes.index(adjacent_nodes[i][0])][2] = gx
                            openlist[openlist_nodes.index(adjacent_nodes[i][0])][3] = fx
            else:
                for i in range(len(list(adjacent_nodes))):
                    gx = nmin[2] + adjacent_nodes[i][1]
                    fx = gx + sim_cost(nodes_pos[adjacent_nodes[i][0]], nodes_pos[end_node])
                    openlist.append([adjacent_nodes[i][0], nmin[0], gx, fx])
        openlist.sort(key=lambda x: x[3])
    # 输出最短路径点集合和最短路径距离
    if openlist and openlist[0][0] == end_node:
        closelist.append(openlist[0])
        cur_nodes = list(list(zip(*list(closelist)))[0])
        father_nodes = list(list(zip(*list(closelist)))[1])
        optimal_path_nodes = [cur_nodes[-1]]
        # costs = [closelist[-1][2]]
        change_index = -1
        while optimal_path_nodes[-1] != start_node:
            if father_nodes[change_index] in cur_nodes:
                # 加入新节点
                optimal_path_nodes.append(cur_nodes[cur_nodes.index(father_nodes[change_index])])
                # costs.append(closelist[cur_nodes.index(father_nodes[change_index])][2])
                change_index = cur_nodes.index(father_nodes[change_index])
        optimal_path_nodes.reverse()
        # costs.reverse()
        return round(openlist[0][2], 2), optimal_path_nodes
    if not openlist:
        return -1, []


def read_trace_info():
    path = 'c:/Users/RTX2080Ti/Desktop/simply_traces.csv'
    data = pd.read_csv(path, sep=',')
    df = pd.DataFrame(data, columns=["feature_index", "feature_pts", "pts_distance"])
    feature_index_data, feature_pts_data, pts_distance_data = np.array(df.feature_index), np.array(
        df.feature_pts), np.array(df.pts_distance)

    # 字典列表
    all_feature_pts, all_pts_distance = [], []
    for i in range(len(feature_index_data)):
        # 生成字典
        try:
            feature_pts, pts_distance = dict(), dict()
            for j in range(len(feature_index_data[i].split(","))):
                feature_pts[int(feature_index_data[i].split(",")[j])] = [float(feature_pts_data[i].split(",")[j * 2]),
                                                                         float(
                                                                             feature_pts_data[i].split(",")[j * 2 + 1])]
                pts_distance[int(feature_index_data[i].split(",")[j])] = float(pts_distance_data[i].split(",")[j])
            all_feature_pts.append(feature_pts)
            all_pts_distance.append(pts_distance)
        except:
            pass
    return all_feature_pts, all_pts_distance


# 获取轨迹特征点的匹配节点集合
# 若没有匹配到，2种情况：阈值设置小了或者是假特征点。
def trace_pts_match_nodes():
    # 利用sklearn输出轨迹转弯特征点匹配到的路网节点集合
    all_feature_adj_pts = []
    # 引入sklearn的kd树求取阈值范围内的道路交叉点nodes,每条轨迹返回一个候选节点集合。
    for i in range(len(all_feature_pts)):
        # 可能需要修改的地方1：动态设定阈值，以免候选路网节点太多。
        dist_threshold = 70 / (2 * np.pi * 6371009) * 360
        data = copy.deepcopy(nodes)
        # len(nodes) 611
        for j in range(len(all_feature_pts[i])):
            data.append(list(all_feature_pts[i].values())[j])
        # print(len(data), data[-(len(all_feature_pts[i])):])
        data = np.array(data)
        # 调包
        tree = BallTree(data, metric='haversine')
        idx = tree.query_radius(data, r=dist_threshold)

        # tree = KDTree(data, metric='euclidean')
        # idx = tree.query_radius(data, r=dist_threshold)
        adj_idx_list = idx[-(len(all_feature_pts[i])):]
        feature_pts_idx = [len(nodes) + i for i in range(len(all_feature_pts[i]))]
        # 匹配不到路口特征点的话，说明是伪特征点，修改all_pts_distance，输出正确的all_feature_adj_pts
        feature_adj_pts = []
        candidate_distance = 0
        delete = 0
        offset = 0
        #
        # print(i, all_pts_distance[i])
        # print(i, [list(i) for i in adj_idx_list])

        for j in range(len(adj_idx_list)):
            adj_pt = []
            for k in range(len(adj_idx_list[j])):
                if adj_idx_list[j][k] not in feature_pts_idx:
                    # if feature_adj_pts and int(adj_idx_list[j][k]) not in feature_adj_pts[-1]:
                    adj_pt.append(int(adj_idx_list[j][k]))
            feature_idx = list(all_feature_pts[i].items())[j][0]
            if adj_pt:
                # 若邻接点不为空，如果前面的特征点匹配的邻接点是空的(delete > 0)，插入该邻接点，修改距离
                if delete != 0:
                    all_pts_distance[i][feature_idx] += candidate_distance
                    candidate_distance = 0
                    delete = 0
                if feature_adj_pts and delete == 0:
                    # 若邻接点不为空，且前面的特征点匹配的邻接点非空
                    # 先判断adj_pt是否等于feature_adj_pts[-1]，若等于，则执行以下操作。
                    # 若不等于，进一步判断现在的邻接点集合中是否有节点位于前面的特征点匹配的邻接点集合中，
                    # 若无，feature_adj_pts加入adj_pt；若有，去除adj_pt中的这些点，例如：adj_pt == [2], feature_adj_pts[-1] == [1,2]
                    # 之后若adj_pt非空，feature_adj_pts加入adj_pt;若adj_pt为空，则执行以下操作。
                    # 执行操作：删除上一个邻接点集合，插入adj_pt，修改all_pts_distance的距离;修改all_pts_distance的上一项的距离，删除本项（feature_idx）
                    old_adj_pt = copy.deepcopy(adj_pt)
                    if adj_pt != feature_adj_pts[-1]:
                        recur_node = [i for i in adj_pt if i in feature_adj_pts[-1]]
                        if recur_node:
                            for node in recur_node:
                                adj_pt.remove(node)
                    if old_adj_pt == feature_adj_pts[-1] or not adj_pt:
                        # print(77777, feature_idx, all_pts_distance[i])
                        sub_keys = list(all_pts_distance[i].keys())
                        prev_feature_idx = sub_keys[sub_keys.index(feature_idx) - 1]
                        prev_distance = all_pts_distance[i][prev_feature_idx]
                        # 插入该邻接点，修改距离
                        all_pts_distance[i][feature_idx] += prev_distance
                        # print(77777, prev_feature_idx, all_pts_distance[i][prev_feature_idx])
                        # 删除上一个邻接点集合，删除all_pts_distance中上一个特征点对应项
                        del all_pts_distance[i][prev_feature_idx]
                        # 若old_adj_pt == feature_adj_pts[-1]，则置零adj_pt
                        adj_pt = []
                        offset += 1
                if adj_pt:
                    feature_adj_pts.append(adj_pt)
            # 邻接点为空，判定为假特征点，修改all_pts_distance对应项，且feature_adj_pts不加入任何项
            else:
                distance = list(all_pts_distance[i].items())[j - offset][1]
                candidate_distance += distance
                del all_pts_distance[i][feature_idx]
                delete = 1
                offset += 1
        all_feature_adj_pts.append(feature_adj_pts)
        # print(i, len(all_pts_distance[i]), all_pts_distance[i], sum(list(all_pts_distance[i].values())))
        print(i, len(feature_adj_pts), feature_adj_pts)
        # print()
    # print(all_feature_adj_pts)
    print("match_trace_pts_nodes_done")
    return all_feature_adj_pts


# 根据候选节点集合生成候选子路径集合，再生成路径，完成地图匹配
def match_all_paths():
    all_match_paths = []
    error_count = 0
    for i in range(len(all_feature_adj_pts)):
        sub_paths_list = []
        feature_adj_pts = all_feature_adj_pts[i]
        pts_distance = all_pts_distance[i]
        for j in range(1, len(pts_distance)):
            sub_paths = generate_sub_path(list(pts_distance.values())[j], feature_adj_pts[j - 1],
                                          feature_adj_pts[j])
            sub_paths_list.append(sub_paths)

        # 输出验证
        # print()
        # print(i, "sub_paths_list", sub_paths_list)
        # print(i, "feature_adj_pts", feature_adj_pts)
        # print(i, pts_distance)
        # cal_cnt = 0
        # for j in range(len(sub_paths_list)):
        #     print(len(sub_paths_list[j]), sub_paths_list[j])
        # for j in range(1, len(sub_paths_list)):
        #     cal_cnt += len(sub_paths_list[j - 1]) * len(sub_paths_list[j])
        # print(i, "cal_cnt", int(cal_cnt))

        # 轨迹推算时，由前到后依次推算路径。最后的候选路径一般是1条，但也可能会超过1条
        # 扩展，是否需要剪枝？
        # 部分轨迹的求解时间过长，如何进一步优化？
        if sub_paths_list:
            match_paths = sub_paths_list[0]
            j = 1
            while j < len(sub_paths_list):
                new_match_paths = copy.deepcopy(match_paths)
                match_paths = []
                for path in new_match_paths:
                    for path1 in sub_paths_list[j]:
                        if path[1][-1] == path1[1][0]:
                            if path[1] + path1[1][1:] not in list(list(zip(*new_match_paths))[1]):
                                match_paths += [[path[0] + path1[0], path[1] + path1[1][1:]]]
                j += 1
                # print(match_paths)
            # 选出一条最优的候选路径
            match_paths.sort(key=lambda x: abs(x[0] - sum(list(pts_distance.values()))))
            all_match_paths.append(match_paths[0])
    return all_match_paths


def generate_sub_path(dist_threshold, adj_pts1, adj_pts2):
    sub_paths = []
    # 加入时间限制
    for i in range(len(adj_pts1)):
        for j in range(len(adj_pts2)):
            cost, optimal_path_nodes = Astar(init_graph, nodes, adj_pts1[i], adj_pts2[j])
            # print(cost, optimal_path_nodes)
            # if abs(cost - dist_threshold) < 1000:
            sub_paths.append([cost, optimal_path_nodes])
    return sub_paths


def trace_matching():
    # 邻接点与路段的映射，统计每条边通过的轨迹个数。
    # 一个隐含的坑：如果轨迹通过一个路段2次，怎么处理？
    # 和人工匹配的结果存在一定误差。
    edge_idx = [i for i in range(len(edges_info))]
    all_trace_cnts = [0 for _ in range(len(edges_info))]
    trace_cnts = dict(zip(edge_idx, all_trace_cnts))
    for i in range(len(all_match_paths)):
        path = all_match_paths[i][1]
        match_edges = []
        for j in range(len(path) - 1):
            cur_node = path[j]
            next_node = path[j + 1]
            sub_graph = init_graph[cur_node]
            next_node_idx = list(list(zip(*sub_graph))[0]).index(next_node)
            match_edge = list(list(zip(*sub_graph))[1])[next_node_idx]
            match_edges.append(match_edge)
        # print(match_edges)
        match_edges = list(set(match_edges))
        for j in range(len(match_edges)):
            trace_cnts[match_edges[j]] += 1
    return trace_cnts


def generate_empirical_distribution():
    trace_cnts_sum = sum(list(list(zip(*list(trace_cnts.items())))[1]))
    empirical_distribution = copy.deepcopy(trace_cnts)
    for i, e in empirical_distribution.items():
        empirical_distribution[i] /= trace_cnts_sum
    print(empirical_distribution)
    df = pd.DataFrame({'edge': list(empirical_distribution.keys()), 'trace_cnts': list(trace_cnts.values()),
                       'probability': list(empirical_distribution.values())})
    save_path1 = "c:/Users/RTX2080Ti/Desktop/经验概率分布" + str(start_node) + "-" + str(end_node) + ".xlsx"
    df.to_excel(save_path1)


def generate_empirical_distribution1():
    trace_node_cnts = dict()
    for i in range(len(all_match_paths)):
        path = all_match_paths[i][1]
        for j in range(len(path)):
            if path[j] not in trace_node_cnts:
                trace_node_cnts[path[j]] = 1
            else:
                trace_node_cnts[path[j]] += 1
    trace_node_cnts[start_node] = len(all_match_paths)
    trace_node_cnts[end_node] = len(all_match_paths)
    node_cnts_sum = sum(list(list(zip(*list(trace_node_cnts.items())))[1]))
    empirical_distribution = copy.deepcopy(trace_node_cnts)
    for i, e in empirical_distribution.items():
        empirical_distribution[i] /= node_cnts_sum
    df = pd.DataFrame({'node': list(trace_node_cnts.keys()), 'trace_cnts': list(trace_node_cnts.values()),
                       'probability': list(empirical_distribution.values())})
    save_path1 = "c:/Users/RTX2080Ti/Desktop/点经验概率分布-" + str(start_node) + "-" + str(end_node) + ".xlsx"
    df.to_excel(save_path1)


if __name__ == '__main__':
    # 根据Arcmap空间连接工具生成的邻接信息表，生成节点的邻接列表（列表的每项按照节点索引升序排列）。
    nodes, startpts, midpts, endpts, edges_info = read_txt()
    init_graph, edge_count = generate_adjacent_list()

    # 简单的图结构验证
    # nx_graph = nx.DiGraph()
    # node_list = [i for i in range(len(list(init_graph)))]
    # nx_graph.add_nodes_from(node_list)
    # all_weight_edges = []
    # for i in range(len(list(init_graph))):
    #     for j in range(len(list(init_graph[i]))):
    #         # 节点，边的邻接节点，邻接边，最小时间成本
    #         all_weight_edges.append((i, init_graph[i][j][0]))
    # nx_graph.add_edges_from(all_weight_edges)
    # # print('all_weight_edges', all_weight_edges)
    # print('图中节点的个数', nx_graph.number_of_nodes())
    # print('图中边的个数', nx_graph.number_of_edges())

    start_t = datetime.datetime.now()
    # 路网匹配主流程：400多条轨迹2.84秒匹配完毕，速度完全满足预期。
    all_feature_pts, all_pts_distance = read_trace_info()
    all_feature_adj_pts = trace_pts_match_nodes()
    all_match_paths = match_all_paths()

    end_t = datetime.datetime.now()
    sec = (end_t - start_t).total_seconds()
    print("所用时间", sec)

    # 应用场景1：适用于经验访问概率生成
    start_node = 83
    end_node = 17
    trace_cnts = trace_matching()
    # generate_empirical_distribution()
    generate_empirical_distribution1()

    exit()
