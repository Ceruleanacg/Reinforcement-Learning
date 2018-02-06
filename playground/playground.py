# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

from helpers import json_helper


if __name__ == "__main__":

    data_a3c = json_helper.load_json("../algorithms/A3C/data/rewards.json")
    data_pg = json_helper.load_json("../algorithms/Policy-Gradient/data/rewards.json")

    plt.plot(np.arange(len(data_a3c)), data_a3c, label="A3C")
    plt.plot(np.arange(len(data_pg)), data_pg, label="Actor-Only")
    plt.title('A3C and Actor-Only for CartPole')
    plt.xlabel('Steps')
    plt.ylabel('Total moving average rewards')
    plt.legend(loc='upper left')
    plt.show()
