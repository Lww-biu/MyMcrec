import argparse
import os
import time

import numpy as np

root = os.path.dirname(os.path.realpath(__file__)) + "/Data/Movielens/"
feature_file_dict = {
    "u": root + "user_node_emb.dat",
    "m": root + "movie_node_emb.dat",
    "g": root + "genre_node_emb.dat",
}
features = {}


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


def get_sim(u, v):
    return u.dot(v) / ((u.dot(u)**0.5) * (v.dot(v)**0.5))


def cal_path_score(paths):
    paths_score = []
    for path in paths:
        score = []
        nodes = path.split("-")
        for i in range(len(nodes) - 1):
            t1 = nodes[i][0]
            t2 = nodes[i + 1][0]
            e1 = int(nodes[i][1:])
            e2 = int(nodes[i + 1][1:])
            f1 = features[t1][e1]
            f2 = features[t2][e2]
            score.append(get_sim(f1, f2))
        paths_score.append((path, np.mean(score)))
    return paths_score


def split_one_pair_path(one_pair_path):
    meta_path = {}
    u_id, m_id = one_pair_path[0].split(",")
    meta_path_pair = [u_id, m_id]
    for path in one_pair_path[2:]:
        path = path.replace("\n", "")
        nodes = path.split("-")
        meta_type = ""
        for node in nodes:
            node_type = node[0]
            meta_type += node_type
        if meta_type in meta_path:
            meta_path[meta_type].append(path)
        else:
            meta_path.update({meta_type: [path]})

    meta_path_pair.append(meta_path)

    return meta_path_pair


def load_path(fr_path, sample_size):
    for line in fr_path:
        lines = line.split("\t")
        meta_path_pair = split_one_pair_path(lines)
        u_id, m_id, meta_path = meta_path_pair
        for path_type in meta_path:
            arr = u_id + "," + m_id + "\t"
            paths = meta_path[path_type]
            # 计算paths中每条path的优先级
            paths = cal_path_score(paths)
            path_size = len(paths)
            paths.sort(key=lambda x: x[1], reverse=True)

            if path_size > sample_size:
                paths = paths[:sample_size]

            arr += str(len(paths)) + "\t"

            file_name = root + "meta-path/" + path_type + ".dat"
            f = open(file_name, "a", encoding="utf-8")
            for path in paths:
                arr = arr + path[0] + " "
            arr = arr.strip(" ") + "\n"
            f.write(arr)
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""change the format of the path data""")

    parser.add_argument(
        "--path",
        type=str,
        dest="path_file",
        default=root + "paths.dat",
    )
    parser.add_argument(
        "--samplesize",
        type=int,
        dest="sample_size",
        default=5,
        help=
        "the sampled size of paths between nodes with choices [5, 10, 20, ...]",
    )

    parsed_args = parser.parse_args()

    path_file = parsed_args.path_file
    sample_size = parsed_args.sample_size

    fr_path = open(path_file, "r", encoding="utf-8")

    print("---------------Split Path---------------")
    start_time = time.perf_counter()

    load_feature_as_map(feature_file_dict)
    load_path(fr_path, sample_size)

    print("Split Path %.2f seconds" % (time.perf_counter() - start_time))

    fr_path.close()
