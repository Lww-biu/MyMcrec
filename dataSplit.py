import argparse
import os
import time

root = os.path.dirname(os.path.realpath(__file__)) + "/Data/Yelp/"


def round_int(rating_num, ratio):

    train_size = int(round(rating_num * ratio, 0))

    return train_size


def load_rating(fr_rating):
    rating_data = {}

    for line in fr_rating:
        lines = line.split("\t")
        user, item = lines[0], lines[1]
        if user in rating_data:
            rating_data[user].append(item)
        else:
            rating_data.update({user: [item]})
    rating_data = sorted(rating_data.items(), key=lambda x: int(x[0]))
    rating_data = {k: v for k, v in rating_data}
    return rating_data


def split_rating_into_train_test(rating_data, fw_train, fw_test, ratio):
    for user in rating_data:
        item_list = rating_data[user]
        rating_num = rating_data[user].__len__()
        train_size = round_int(rating_num, ratio)

        test_line = user + ","

        flag = 0

        for item in item_list:
            if flag < train_size:
                line = user + " " + item + "\n"
                fw_train.write(line)
                flag = flag + 1
            else:
                test_line = test_line + item + " "

        test_line = test_line.strip(' ') + "\n"
        if train_size == rating_num:
            continue
        else:
            fw_test.write(test_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""change the format of the path data""")

    parser.add_argument(
        "--dataset",
        type=str,
        dest="dataset",
        default="Movielens",
    )
    parser.add_argument(
        "--rating",
        type=str,
        dest="rating_file",
        default=root + "user_business.dat",
    )
    parser.add_argument(
        "--train",
        type=str,
        dest="train_file",
        default=root + "train.dat",
    )
    parser.add_argument(
        "--test",
        type=str,
        dest="test_file",
        default=root + "test.dat",
    )
    parser.add_argument("--ratio", type=float, dest="ratio", default=0.8)

    parsed_args = parser.parse_args()

    dataset = parsed_args.dataset
    rating_file = parsed_args.rating_file
    train_file = parsed_args.train_file
    test_file = parsed_args.test_file
    ratio = parsed_args.ratio

    fr_rating = open(rating_file, "r", encoding="utf-8")
    fw_train = open(train_file, "w", encoding="utf-8")
    fw_test = open(test_file, "w", encoding="utf-8")

    print("---------------Split Data---------------")
    start_time = time.perf_counter()

    rating_data = load_rating(fr_rating)
    split_rating_into_train_test(rating_data, fw_train, fw_test, ratio)

    print("---------Split Data %.2f seconds---------" %
          (time.perf_counter() - start_time))

    fr_rating.close()
    fw_train.close()
    fw_test.close()
