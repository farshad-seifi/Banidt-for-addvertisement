import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

# Test data for finding optimal policy
dataset = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\RL\Bandit\Ads_Optimisation.csv")

# Epsilon value for epsilon greedy policy
eps = 0.25


# Random Search Policy
def Random_Search(dataset):

    """
    :param dataset: Test given dataset
    :return: Selected Arm data and gained reward
    """

    # Number of plays
    N = len(dataset)
    # Number of arms
    d = len(dataset.columns)

    # Data frames for selected arm and rewards
    ads_selected = pd.DataFrame(np.zeros((N, 1)))
    reward_gained = pd.DataFrame(np.zeros((N, 1)))


    for n in range(0, N):
        # Selecting an arm randomly
        ad = random.randrange(d)
        ads_selected.iloc[n] = ad
        # Calculate the corresponding reward
        reward_gained.iloc[n] = dataset.values[n, ad]

    # Adding the selected arm and reward to data
    ads_selected.columns = ["Selected_ad"]
    reward_gained.columns = ["Reward"]

    return ads_selected, reward_gained

# Epsilon Greedy Policy
def Epsilon_Greedy(dataset, eps, Stationary, alpha):

    """
    :param dataset: Test given dataset
    :param eps: Value of epsilon
    :param Stationary: Whether the problem is stationary or not
    :param alpha: The discount value for stationary
    :return: Selected Arm data and gained reward
    """

    # Number of plays
    N = len(dataset)
    # Number of arms
    d = len(dataset.columns)

    # Data frames for selected arm and rewards
    ads_selected = pd.DataFrame(np.zeros((N, 1)))
    reward_gained = pd.DataFrame(np.zeros((N, 1)))

    ad_data = pd.DataFrame(np.zeros((d,4)))
    ad_data.columns = ["ad", "count", "reward", "Q"]

    for i in range(0, len(ad_data)):
        ad_data["ad"].iloc[i] = i

    for i in range(0, N):

        if i == 0:
            # Selecting the first arm randomly
            ad = random.randrange(d)
        else:
            # Selecting some arms randomly based on the epsilon value
            if random.random() < eps:
                ad = random.randrange(d)
            else:
                if Stationary:
                    # Selecting next arm based on Q value in stationary situation
                    ad = np.argmax(ad_data["reward"] / ad_data["count"])
                else:
                    # Selecting next arm based on Q value in stationary non-situation
                    ad = np.argmax(ad_data["Q"])


        ad_data["count"].iloc[ad] += 1
        # Updating Q value
        ad_data["reward"].iloc[ad] += dataset.values[i, ad]
        if not  Stationary:
            ad_data["Q"].iloc[ad] = ad_data["Q"].iloc[ad] + (1/alpha) * (dataset.values[i, ad] - ad_data["Q"].iloc[ad])
        ads_selected.iloc[i] = ad
        reward_gained.iloc[i] = dataset.values[i, ad]

    ads_selected.columns = ["Selected_ad"]
    reward_gained.columns = ["Reward"]

    return ads_selected, reward_gained, ad_data


# Softmax Policy
def Softmax(dataset):

    """
    :param dataset: Test given dataset
    :return: Selected Arm data and gained reward
    """

    # Number of plays
    N = len(dataset)
    # Number of arms
    d = len(dataset.columns)

    # Data frames for selected arm and rewards
    ads_selected = pd.DataFrame(np.zeros((N, 1)))
    reward_gained = pd.DataFrame(np.zeros((N, 1)))

    ad_data = pd.DataFrame(np.zeros((d,3)))
    ad_data.columns = ["ad", "count", "reward"]

    for i in range(0, len(ad_data)):
        ad_data["ad"].iloc[i] = i

    for i in range(0, N):
        # Selecting first arm randomly
        if i == 0:
            ad = random.randrange(d)
        else:
            # Selecting next arms based on softmax function
            up = np.exp(ad_data["reward"] / ad_data["count"])
            up = [0.5 if math.isnan(x) else x for x in up]
            up = np.array(up)
            probability_vec = up/up.sum()
            ad = np.random.choice(list(ad_data["ad"]), 1 , p = list(probability_vec))[0]
            ad = np.int(ad)

        # Calculating the value of reward
        ad_data["count"].iloc[ad] += 1
        ad_data["reward"].iloc[ad] += dataset.values[i, ad]
        ads_selected.iloc[i] = ad
        reward_gained.iloc[i] = dataset.values[i, ad]

    ads_selected.columns = ["Selected_ad"]
    reward_gained.columns = ["Reward"]

    return ads_selected, reward_gained


# UCB Policy
def UCB(dataset):


    """
    :param dataset: Test given dataset
    :return: Selected Arm data and gained reward
    """

    # Number of plays
    N = len(dataset)
    # Number of arms
    d = len(dataset.columns)

    # Data frames for selected arm and rewards
    ads_selected = pd.DataFrame(np.zeros((N, 1)))
    reward_gained = pd.DataFrame(np.zeros((N, 1)))

    ad_data = pd.DataFrame(np.zeros((d,3)))
    ad_data.columns = ["ad", "count", "reward"]

    for i in range(0, len(ad_data)):
        ad_data["ad"].iloc[i] = i

    for i in range(0, N):
        # Selecting all the arms at least one time
        if i < d:
            ad = i
        # Selecting all other arms based on Q value
        else:
            ad = np.argmax((ad_data["reward"] / ad_data["count"]) + np.sqrt(2*(math.log10(i)) / ad_data["count"]))


        ad_data["count"].iloc[ad] += 1
        ad_data["reward"].iloc[ad] += dataset.values[i, ad]
        ads_selected.iloc[i] = ad
        reward_gained.iloc[i] = dataset.values[i, ad]

    ads_selected.columns = ["Selected_ad"]
    reward_gained.columns = ["Reward"]

    return ads_selected, reward_gained


# Reward Calculator
def Reward_Calculator(reward_gained):

    """
    :param reward_gained: Reward data
    :return: It calculates average of reward over time and total of it
    """
    N = len(dataset)
    average_reward_gained = pd.DataFrame(np.zeros((N, 1)))
    total_reward_gained = pd.DataFrame(np.zeros((N, 1)))

    average_reward_gained.iloc[0] = reward_gained.iloc[0]
    total_reward_gained.iloc[0] = reward_gained.iloc[0]

    for i in range(1, N):
        average_reward_gained.iloc[i] = reward_gained[:i].mean()
        total_reward_gained.iloc[i] = reward_gained[:i].sum()


    return average_reward_gained, total_reward_gained


def Optimium_Policy(dataset):

    """
    :param dataset: Main Dataset
    :return: Maximum possible reward based on selecting best arm each time
    """
    N = len(dataset)
    ads_selected = pd.DataFrame(np.zeros((N, 1)))
    reward_gained = pd.DataFrame(np.zeros((N, 1)))

    for i in range(0, len(dataset)):
        ads_selected.iloc[i] = dataset.iloc[i].argmax()
        reward_gained.iloc[i] = dataset.iloc[i].max()

    return ads_selected, reward_gained



# Runing different scenarios
ads_selected_UCB, reward_gained_UCB = UCB(dataset)
ads_selected_Random, reward_gained_Random = Random_Search(dataset)
ads_selected_Epsilon_01, reward_gained_Epsilon_01, ad_data_01 = Epsilon_Greedy(dataset, 0.1, True, 10)
ads_selected_Epsilon_02, reward_gained_Epsilon_02, ad_data_02 = Epsilon_Greedy(dataset, 0.2, True, 10)
ads_selected_Softmax, reward_gained_Softmax = Softmax(dataset)
ads_selected_optimum, reward_gained_optimum = Optimium_Policy(dataset)


average_reward_gained_UCB, total_reward_gained_UCB = Reward_Calculator(reward_gained_UCB)
average_reward_gained_Random, total_reward_gained_Random = Reward_Calculator(reward_gained_Random)
average_reward_gained_Softmax, total_reward_gained_Softmax = Reward_Calculator(reward_gained_Softmax)
average_reward_gained_Epsilon_01, total_reward_gained_Epsilon_01 = Reward_Calculator(reward_gained_Epsilon_01)
average_reward_gained_Epsilon_02, total_reward_gained_Epsilon_02 = Reward_Calculator(reward_gained_Epsilon_02)
average_reward_gained_optimum, total_reward_gained_optimum = Reward_Calculator(reward_gained_optimum)

# Depicting the reward figure
plt.plot(average_reward_gained_UCB[10:], label = "UCB Search Policy")
plt.plot(average_reward_gained_Random[10:], label = "Random Search Policy")
plt.plot(average_reward_gained_Softmax[10:], label = "Softmax Policy")
plt.plot(average_reward_gained_Epsilon_01[10:], label = "Epsilon Greedy Policy eps 0.1")
plt.plot(average_reward_gained_Epsilon_02[10:], label = "Epsilon Greedy Policy eps 0.2")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.title("Different Policy Average Reward")
plt.legend(loc='lower right')
plt.ylim([0,0.35])
plt.show()

# Depicting the regret figure
plt.plot((average_reward_gained_optimum-average_reward_gained_UCB)[10:], label = "UCB Search Policy")
plt.plot((average_reward_gained_optimum-average_reward_gained_Random)[10:], label = "Random Search Policy")
plt.plot((average_reward_gained_optimum-average_reward_gained_Softmax)[10:], label = "Softmax Policy")
plt.plot((average_reward_gained_optimum-average_reward_gained_Epsilon_01)[10:], label = "Epsilon Greedy Policy eps 0.1")
plt.plot((average_reward_gained_optimum-average_reward_gained_Epsilon_02)[10:], label = "Epsilon Greedy Policy eps 0.2")
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.title("Different Policy Average Regret")
plt.legend(loc='lower right')
plt.ylim([0,1])
plt.show()

