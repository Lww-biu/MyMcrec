import argparse
import os
from random import randint
import time

root = os.path.dirname(os.path.realpath(__file__)) + "/Data/Yelp/"


def load_data(fr_rating, fr_test):
    rating_data = {}
    all_movie_list = []

    for line in fr_rating:
        lines = line.split("\t")
        user = lines[0]
        movie = lines[1].replace("\n", "")

        if user in rating_data:
            rating_data[user].append(movie)
        else:
            rating_data.update({user: [movie]})

        # rating_data = sorted(rating_data.items(), key=lambda x: int(x[0]))
        # rating_data = {k: v for k, v in rating_data}

        if movie not in all_movie_list:
            all_movie_list.append(movie)

    all_movie_list.sort(key=lambda x: int(x))

    user_test_movie_size = {}
    for line in fr_test:
        lines = line.split(",")
        user = lines[0]
        test_dict = lines[1].split(" ")
        test_dict_size = len(test_dict)
        user_test_movie_size.update({user: test_dict_size})

    return rating_data, all_movie_list, user_test_movie_size


def negative_sample(rating_dict, all_movie_list, user_test_movie_size, ratio,
                    fw_negative):
    all_movie_size = len(all_movie_list)

    for user in rating_dict:
        user_rating_movie = rating_dict[user]
        # user_test_size = user_test_movie_size[user]

        # user_rating_movie_size = len(user_rating_movie)
        # negative_size = min(user_test_size * ratio,
        #                     all_movie_size - user_rating_movie_size)
        user_negative_movie = []

        for i in range(all_movie_size):
            negative_movie = str(all_movie_list[i])
            if (negative_movie not in user_rating_movie
                    and negative_movie not in user_negative_movie):
                user_negative_movie.append(negative_movie)

        # while len(user_negative_movie) < negative_size:
        #     negative_index = randint(0, (all_movie_size - 1))
        #     negative_movie = str(all_movie_list[negative_index])
        #     if (negative_movie not in user_rating_movie
        #             and negative_movie not in user_negative_movie):
        #         user_negative_movie.append(negative_movie)

        line = user + "," + " ".join(user_negative_movie) + "\n"
        fw_negative.write(line)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=""" Sample Negative Movies for Each User""")

    parser.add_argument("--rating",
                        type=str,
                        dest="rating_file",
                        default=root + "user_business.dat")
    parser.add_argument("--test_file",
                        type=str,
                        dest="test_file",
                        default=root + "test.dat")
    parser.add_argument("--negative",
                        type=str,
                        dest="negative_file",
                        default=root + "negative.dat")
    parser.add_argument("--negative_ratio",
                        type=float,
                        dest="ratio",
                        default=20)

    parsed_args = parser.parse_args()

    rating_file = parsed_args.rating_file
    test_file = parsed_args.test_file
    negative_file = parsed_args.negative_file
    ratio = parsed_args.ratio

    fr_rating = open(rating_file, "r")
    fr_test = open(test_file, "r")
    fw_negative = open(negative_file, "w")

    print("---------------Negative Sample---------------")
    start_time = time.perf_counter()

    rating_dict, all_movie_list, user_test_movie_size = load_data(
        fr_rating, fr_test)

    negative_sample(rating_dict, all_movie_list, user_test_movie_size, ratio,
                    fw_negative)

    print("Negative Sample %.2f seconds" % (time.perf_counter() - start_time))

    fr_rating.close()
    fw_negative.close()
