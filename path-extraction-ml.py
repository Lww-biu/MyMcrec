import argparse
from json import load
import os
import random
import time
import networkx as nx
import numpy as np

# ml 100k
usize = 943 + 1
msize = 1682 + 1
tsize = 18 + 1
root = os.path.dirname(os.path.realpath(__file__)) + "/Data/Movielens/"
feature_file_dict = {
    "u": root + "user_node_emb2.dat",
    "m": root + "item_node_emb2.dat",
    "g": root + "genre_node_emb2.dat",
}
features = {}


def load_data(file):
    data = []

    for line in file:
        lines = line.split(" ")
        user, movie = lines[0], lines[1].replace("\n", "")
        data.append((user, movie))

    return data


def load_feature_as_map(feature_file_dict):

    global features
    features = {}
    node_sizes = {}
    feature_size = -1
    for feature_type, feature_file_path in feature_file_dict.items():
        with open(feature_file_path) as input:
            count = 0
            for line in input.read().splitlines():
                line = line.strip()
                if line == "":
                    continue
                count += 1
                arr = line.split(",")
                if feature_size == -1:
                    # arr第一位是id，所以特征大小为len(arr) - 1
                    feature_size = len(arr) - 1
                else:
                    assert feature_size == (len(arr) - 1)

            node_sizes[feature_type] = count

    for feature_type, feature_file_path in feature_file_dict.items():
        features[feature_type] = np.zeros(
            (node_sizes[feature_type] + 1, feature_size), dtype=np.float32)
        with open(feature_file_path) as input:
            for line in input.readlines():
                line = line.strip()
                if line == "":
                    continue

                arr = line.strip().split(",")
                node_id = int(arr[0])

                for j in range(len(arr[1:])):
                    features[feature_type][node_id][j] = float(arr[j + 1])
    return features


def add_user_movie_interaction_into_graph(rating_data):
    Graph = nx.DiGraph()

    for pair in rating_data:
        user, movie = pair[0], pair[1]
        user_node = "u" + user
        movie_node = "m" + movie
        Graph.add_node(user_node)
        Graph.add_node(movie_node)
        Graph.add_edge(user_node, movie_node)

    return Graph


def add_genre_into_graph(fr_genre, Graph):
    for line in fr_genre:
        lines = line.replace("\n", "").split("\t")
        movie_id, genre_id = lines[0], lines[1]

        movie_node = "m" + movie_id
        if not Graph.has_node(movie_node):
            Graph.add_node(movie_node)

        genre_node = "g" + genre_id
        if not Graph.has_node(genre_node):
            Graph.add_node(genre_node)
        Graph.add_edge(movie_node, genre_node)
        Graph.add_edge(genre_node, movie_node)

    return Graph


def load_knn(fr_knn):
    ddict = {}
    for line in fr_knn:
        lines = line.split("\t")
        id1 = lines[0]
        id2 = lines[1]
        score = lines[2].replace("\n", "")
        if id1 in ddict:
            ddict[id1].update({id2: score})
        else:
            ddict.update({id1: {id2: score}})
    return ddict


def add_user_user_knn_into_graph(fr_user_knn, sim_size, Graph):
    uu_dict = load_knn(fr_user_knn)

    for u_id in uu_dict:
        uu_list = sorted(uu_dict[u_id].items(),
                         key=lambda s: s[1],
                         reverse=True)[:sim_size]
        uu_dict[u_id] = uu_list

    for u_id1 in uu_dict:
        tmp_list = uu_dict[u_id1]
        for i in range(len(tmp_list)):
            u_id2 = tmp_list[i][0]
            Graph.add_edge("u" + u_id1, "u" + u_id2)
            Graph.add_edge("u" + u_id2, "u" + u_id2)
    return Graph


def add_movie_movie_knn_into_graph(fr_movie_knn, sim_size, Graph):
    mm_dict = load_knn(fr_movie_knn)

    for m_id in mm_dict:
        mm_list = sorted(mm_dict[m_id].items(),
                         key=lambda s: s[1],
                         reverse=True)[:sim_size]
        mm_dict[m_id] = mm_list

    for m_id1 in mm_dict:
        tmp_list = mm_dict[m_id1]
        for i in range(len(tmp_list)):
            m_id2 = tmp_list[i][0]
            Graph.add_edge("m" + m_id1, "m" + m_id2)
            Graph.add_edge("m" + m_id2, "m" + m_id2)
    return Graph


def print_graph_statistic(Graph):
    """
    output the statistic info of the graph

    Input:
        @Graph: the built graph
    """
    print("The knowledge graph has been built completely \n")
    print("The number of nodes is:  " + str(len(Graph.nodes())) + " \n")
    print("The number of edges is  " + str(len(Graph.edges())) + " \n")


def get_sim(u, v):
    return u.dot(v) / ((u.dot(u)**0.5) * (v.dot(v)**0.5))


# def cal_path_score(paths):
#     # paths_score = []
#     # for path in paths:
#     #     t1 = path[0][0]
#     #     t2 = path[1][0]
#     #     e1 = int(path[0][1:])
#     #     e2 = int(path[1][1:])
#     #     score = get_sim(features[t1][e1], features[t2][e2])
#     #     paths_score.append((path, score))

#     # return paths_score
#     paths_score = []
#     for path in paths:
#         score = []
#         for i in range(len(path) - 1):
#             t1 = path[i][0]
#             t2 = path[i + 1][0]
#             e1 = int(path[i][1:])
#             e2 = int(path[i + 1][1:])
#             f1 = features[t1][e1]
#             f2 = features[t2][e2]
#             score.append(get_sim(f1, f2))
#         paths_score.append((path, np.mean(score)))
#     return paths_score


def mine_paths_between_nodes(Graph, user_node, movie_node, maxlen,
                             sample_size):
    connected_path = []
    for path in nx.all_simple_paths(Graph,
                                    source=user_node,
                                    target=movie_node,
                                    cutoff=maxlen):
        if len(path) == maxlen + 1:
            connected_path.append(path)

    # 计算connected_path中每条path的优先级
    # connected_path = cal_path_score(connected_path)
    # path_size = len(connected_path)
    # connected_path.sort(key=lambda x: x[1], reverse=True)

    # if path_size > sample_size:
    #     connected_path = connected_path[:sample_size]

    return connected_path


def dump_paths(Graph, maxlen, sample_size, fw_file):
    pair_list = []
    # only train
    for u in range(1, usize):
        for i in range(1, msize):
            pair_list.append([u, i])
    for pair in pair_list:
        # for pair in rating_data:
        user_id, movie_id = str(pair[0]), str(pair[1])
        user_node = "u" + user_id
        movie_node = "m" + movie_id
        line = user_id + "," + movie_id + "\t"

        if Graph.has_node(user_node) and Graph.has_node(movie_node):
            one_pair_paths = mine_paths_between_nodes(Graph, user_node,
                                                      movie_node, maxlen,
                                                      sample_size)
            if len(one_pair_paths) == 0:
                continue
            line = line + str(len(one_pair_paths)) + "\t"
            for path in one_pair_paths:
                line = line + "-".join(path) + "\t"

            line = line.strip('\t') + "\n"

            fw_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Build Knowledge Graph and Mine the Connected Paths""")

    parser.add_argument(
        "--train",
        type=str,
        dest="train_file",
        default=root + "train.dat",
    )
    parser.add_argument(
        "--movie_genre",
        type=str,
        dest="genre_file",
        default=root + "movie_genre.dat",
    )
    parser.add_argument(
        "--user_user_knn",
        type=str,
        dest="user_knn_file",
        default=root + "user_user(knn).dat",
    )
    parser.add_argument(
        "--movie_movie_knn",
        type=str,
        dest="movie_knn_file",
        default=root + "movie_movie(knn).dat",
    )
    parser.add_argument(
        "--path",
        type=str,
        dest="path_file",
        default=root + "paths.dat",
    )
    parser.add_argument(
        "--pathlength",
        type=int,
        dest="path_length",
        default=3,
        help="length of paths with choices [3,5,7]",
    )
    parser.add_argument(
        "--samplesize",
        type=int,
        dest="sample_size",
        default=10,
        help=
        "the sampled size of paths between nodes with choices [5, 10, 20, ...]",
    )
    parser.add_argument("--simsize",
                        type=int,
                        dest="sim_size",
                        default=10,
                        help="Number of similar users selected")

    parsed_args = parser.parse_args()

    train_file = parsed_args.train_file
    genre_file = parsed_args.genre_file
    user_knn_file = parsed_args.user_knn_file
    movie_knn_file = parsed_args.movie_knn_file
    path_file = parsed_args.path_file
    path_length = parsed_args.path_length
    sample_size = parsed_args.sample_size
    sim_size = parsed_args.sim_size

    fr_train = open(train_file, "r", encoding="utf-8")
    fr_genre = open(genre_file, "r", encoding="utf-8")
    fr_user_knn = open(user_knn_file, "r", encoding="utf-8")
    fr_movie_knn = open(movie_knn_file, "r", encoding="utf-8")
    fw_path = open(path_file, "w", encoding="utf-8")

    rating_data = load_data(fr_train)
    load_feature_as_map(feature_file_dict)

    # uu_dict = load_uu_knn(fr_user_knn)
    Graph = add_user_movie_interaction_into_graph(rating_data)
    Graph = add_genre_into_graph(fr_genre, Graph)
    Graph = add_user_user_knn_into_graph(fr_user_knn, sim_size, Graph)
    Graph = add_movie_movie_knn_into_graph(fr_movie_knn, sim_size, Graph)
    print_graph_statistic(Graph)

    dump_start = time.perf_counter()
    dump_paths(Graph, path_length, sample_size, fw_path)
    print("dump path %.2f seconds" % (time.perf_counter() - dump_start))
    fr_train.close()
    fr_genre.close()
    fr_user_knn.close()
    fw_path.close()
