import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import ast
from sklearn.preprocessing import LabelEncoder
import math
from scipy.stats import gaussian_kde

My = 2
Mt = 16
n_testing = 10000
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

import torch

from bounds import hb_p_value
from data_utils.dataset import max_seq_length, n_test, n_cals, n_cals1


def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any((costs[:i]) >= c, axis=1)) and np.all(np.any((costs[i + 1:]) >= c, axis=1))
    return is_efficient


def fixed_sequence_testing(h_sorted, p_vals):
    list_rejected = []
    for b in range(len(h_sorted)):
        xx = h_sorted[b]
        if p_vals[xx] < args.delta:
            list_rejected.append(xx + 1)
        else:
            break

    return list_rejected


def discretize_2d_array(array, n_bins=4):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    n_samples, n_features = array.shape
    discretized_array = np.zeros((n_samples, n_features), dtype=int)

    for i in range(n_features):
        discretized_array[:, i] = discretizer.fit_transform(array[:, i].reshape(-1, 1)).flatten()

    return discretized_array



def IB_UCB_YT(loss_vec, delta):
    theta = np.sqrt(2/n_testing*np.log((2**(My*Mt)-2)/delta))
    deltaI = theta/2*np.log2((My*Mt-1)*(My-1)*(Mt-1)) + 3*(-theta/2*np.log2(theta/2)-(1-theta/2)*np.log2(1-theta/2))
    return loss_vec - deltaI

def IB_UCB_XT(loss_vec, delta):
    theta = np.sqrt(2/n_testing*np.log((2**(My*Mt)-2)/delta))
    deltaI = theta/2*np.log2((My*Mt-1)*(My-1)*(Mt-1)) + 3*(-theta/2*np.log2(theta/2)-(1-theta/2)*np.log2(1-theta/2))
    #print(f'delta is {delta} and deltaI is {deltaI} and estimated I is {loss_vec}')
    return loss_vec + deltaI

def IB_p_value_YT(Y, T, alpha, bisec_max_iter=20):
    # we first get the CI per delta
    def _no_intersect(loss_vec, delta):
        curr_UCB = -IB_UCB_YT(loss_vec, delta)  # delta -> u -> internal ..
        return curr_UCB <= -alpha  # no intersection with null

    T_temp = np.array([hash(tuple(x)) for x in T])
    loss_vec = mutual_info_score(Y, T_temp)
    delta = 1
    bi = 1
    for bisec_iter in range(bisec_max_iter):
        bi /= 2
        if _no_intersect(loss_vec, delta):  # no intersection, decrease delta to make it intersect
            delta -= bi
        else:  # intersection, increase delta to make it no intersect
            delta += bi
        # print('bisection delta', delta)
    # print('next delta', delta)
    #print('p value', delta)
    return delta  # p-value

def IB_p_value_XT(X, T, alpha, bisec_max_iter=20):
    # we first get the CI per delta
    le = LabelEncoder()
    def _no_intersect(loss_vec, delta):
        curr_UCB = IB_UCB_XT(loss_vec, delta)  # delta -> u -> internal ..
        return curr_UCB <= alpha  # no intersection with null

    loss_vec = mutual_info_score(X,T)
    delta = 1
    bi = 1
    for bisec_iter in range(bisec_max_iter):
        bi /= 2
        if _no_intersect(loss_vec, delta):  # no intersection, decrease delta to make it intersect
            delta -= bi
        else:  # intersection, increase delta to make it no intersect
            delta += bi
        # print('bisection delta', delta)
    # print('next delta', delta)
    #print('p value', delta)
    return delta  # p-value


def plots(df, alpha):
    results_folder = './results/'
    x = df['YTC']
    x = [j for j in x if math.isfinite(j)]
    x = [i for i in x if not math.isnan(i)]
    x1 = np.min(x)
    x2 = np.max(x)
    x = df['XTC']
    x = [j for j in x if math.isfinite(j)]
    x = [i for i in x if not math.isnan(i)]
    y1 = np.min(x)
    y2 = np.max(x)


    # Create the main plot with subplots for marginals
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['text.usetex'] = True

    # Set up a grid for the plot layout (6x6 grid to accommodate color bars and marginals)
    grid = plt.GridSpec(6, 6, hspace=0.4, wspace=0.4)

    # Marginal distribution along y-axis (left side)
    ax_y_marginal = fig.add_subplot(grid[1:5, 0])
    sns.kdeplot(y=df['XTC'], fill=True, color="blue", ax=ax_y_marginal, label="XTC Marginal")
    sns.kdeplot(y=df['XT'], fill=True, color="red", ax=ax_y_marginal, label="XT Marginal")
    ax_y_marginal.set_xlabel('Density')
    ax_y_marginal.set_ylabel(r'$I_{\lambda^*}(X;T)$')

    # Set the y limits for the y-marginal plot


    # Main KDE plot (central plot)
    ax_main = fig.add_subplot(grid[1:5, 1:5])
    kde1 = sns.kdeplot(x=df['YTC'], y=df['XTC'], cmap='viridis', fill=True, ax=ax_main, cbar=False)
    kde2 = sns.kdeplot(x=df['YT'], y=df['XT'], cmap='Reds', fill=True, ax=ax_main, cbar=False)
    ax_main.axvline(x=alpha, color='red', linestyle='--')

    # Set the x and y limits for the main plot


    # Remove tick labels for the main 2D plot
    ax_main.set_xticklabels([])
    ax_main.set_yticklabels([])
    ax_main.tick_params(left=False, bottom=False)

    # Marginal distribution along x-axis (below the main plot)
    ax_x_marginal = fig.add_subplot(grid[5, 1:5], sharex=None)  # Disable shared x-axis to retain tick labels
    sns.kdeplot(x=df['YTC'], fill=True, color="blue", ax=ax_x_marginal, label="Conventional IB")
    sns.kdeplot(x=df['YT'], fill=True, color="red", ax=ax_x_marginal, label="IB-MHT")
    ax_x_marginal.axvline(x=alpha, color='red', linestyle='--')
    ax_x_marginal.set_xlabel(r'$I_{\lambda^*}(T;Y)$')
    ax_x_marginal.set_ylabel('Density')

    # Set the x limits for the x-marginal plot


    # Ensure x-tick labels are visible on the x-marginal plot
    ax_x_marginal.tick_params(axis='x', labelbottom=True)

    # Get the actual handles and labels from the kde plots in the main axis
    handles, _ = ax_main.get_legend_handles_labels()

    # Add a single combined legend in the top-left corner of the x marginal plot
    ax_x_marginal.legend(handles=handles[:2], labels=['IB-MHT', 'Conventional IB'], loc='upper left', fontsize=12)

    # Adjust the layout to avoid overlap and remove empty spaces
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.tight_layout()

    # Save the figure
    plt.savefig('result_pareto1_with_marginals_and_adjustments.pdf')

    # Show the plot
    plt.show()

    value = alpha

    # Fit the KDE for the x-marginals (you can choose YTC or YT)
    data_YTC = df['YTC']
    data_YT = df['YT']

    # Using scipy's gaussian_kde to fit the KDE
    kde_YTC = gaussian_kde(data_YTC)
    kde_YT = gaussian_kde(data_YT)

    # Define a function to compute the cumulative area up to a given point
    def cumulative_density(kde, value):
        # Create a range of x values to integrate over
        x_vals = np.linspace(min(data_YTC.min(), data_YT.min()), value, 1000)
        # Integrate by approximating the area under the curve up to 'value'
        return np.trapz(kde(x_vals), x_vals)

    # Calculate the area to the left of value for each KDE
    area_YTC = cumulative_density(kde_YTC, value)
    area_YT = cumulative_density(kde_YT, value)

    print(f"Area to the left of {value} for YTC: {area_YTC}")
    print(f"Area to the left of {value} for YT: {area_YT}")

    data_XTC = df['XTC']
    data_XT = df['XT']

    mean_XTC = np.mean(data_XTC)
    std_XTC = np.std(data_XTC)

    print("Mean of conventional:", mean_XTC)
    print("Standard Deviation of conventional:", std_XTC)

    mean_XTC = np.mean(data_XT)
    std_XTC = np.std(data_XT)

    print("Mean of IB-MHT:", mean_XTC)
    print("Standard Deviation of IB-MHT:", std_XTC)



    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 32})
    plt.rcParams['text.usetex'] = True
    sns.kdeplot(x=df['YTC'], y=df['XTC'], cmap='viridis', fill=True, cbar=True)
    sns.kdeplot(x=df['YT'], y=df['XT'], cmap='Reds', fill=True, cbar=True)
    plt.axvline(x=alpha, color='red', linestyle='--', label=f'Target')
    plt.xlabel(r'$I_{\lambda^*}(T;Y)$')
    plt.ylabel(r'$I_{\lambda^*}(X;T)$')
    plt.tight_layout()
    plt.xlim(left = 2.1)
    plt.ylim(bottom = 8.2)
    plt.savefig('result_pareto1.pdf')
    plt.show()


    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 32})
    plt.rcParams['text.usetex'] = True
    sns.kdeplot(x=df['YT'], y=df['XT'], cmap='inferno', fill=True, cbar=False)
    lx = plt.xlim()
    ly = plt.ylim()
    sns.kdeplot(x=df['YTC'], y=df['XTC'], cmap='viridis', fill=True, cbar=True)
    sns.kdeplot(x=df['YT'], y=df['XT'], cmap='inferno', fill=True, cbar=True)
    plt.xlim(lx)
    plt.ylim(ly)
    plt.axvline(x=alpha, color='red', linestyle='--', label=f'Target')
    plt.xlabel(r'$I_{\lambda^*}(T;Y)$')
    plt.ylabel(r'$I_{\lambda^*}(X;T)$')
    plt.tight_layout()
    plt.savefig('result_pareto2.pdf')
    plt.show()



def main(args):
    le = LabelEncoder()
    data_len = args.n_test
    XT_all, YT_all, XT_compare, YT_compare, region_names = [], [], [], [], []
    csv_file_path = 'data_X.csv'
    X = pd.read_csv(csv_file_path)
    X = X.values.tolist()
    X = [tuple(inner_list) for inner_list in X]

    csv_file_path = 'data_Y.csv'
    Y = pd.read_csv(csv_file_path)
    Y = Y.values.tolist()
    Y = [inner_list[0] for inner_list in Y]

    df = pd.read_csv('data_T.csv')
    T_all = []
    # Step 2: Process the 'T' column for each unique BETA
    for b in df['BETA'].unique():
        b_df = df[df['BETA'] == b]

        # Extract the first T string associated with this BETA
        T_string = b_df['T'].values[0]  # Assuming there's only one row per BETA

        # Convert the string representation of T into a list of lists
        T_list_of_lists = ast.literal_eval(T_string)

        # Ensure each inner list has exactly 10 elements and convert each to a tuple
        T = [tuple(inner_list) for inner_list in T_list_of_lists]

        # Check if each tuple has exactly 10 elements
        for i, t in enumerate(T):
            if len(t) != 10:
                raise ValueError(f"Tuple at index {i} does not have 10 elements: {t}")

        T_all.append(T)

    T = T_all

    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)

    flattened_T = T.reshape(-1, T.shape[-1])
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    discretized_values = discretizer.fit_transform(flattened_T)
    T = discretized_values.reshape(T.shape)


    n_cal = args.n_cal
    n_cal1 = args.n_cal1
    n_cal2 = n_cal - n_cal1
    alphas = [float(f) for f in args.alphas.split(',')]
    n_alphas = len(alphas)

    methods = ['Pareto Testing']
    method = methods[0]
    n_methods = len(methods)

    diff_accs = {method: np.zeros((n_alphas, args.n_trials)) for method in methods}
    costs = {method: np.zeros((n_alphas, args.n_trials)) for method in methods}

    for t in tqdm(range(args.n_trials)):

        all_idx = np.arange(data_len)
        np.random.shuffle(all_idx)
        cal_idx = all_idx[:n_cal]
        test_idx = all_idx[n_cal:]

        X_cal = X[cal_idx]
        Y_cal = Y[cal_idx]
        T_cal = T[:, cal_idx] if T is not None else None  # Handle T correctly

        X_cal1 = X_cal[ :n_cal1]
        Y_cal1 = Y_cal[ :n_cal1]
        T_cal1 = T_cal[:, :n_cal1] if T is not None else None

        X_cal2 = X_cal[ n_cal1:]
        Y_cal2 = Y_cal[ n_cal1:]
        T_cal2 = T_cal[:, n_cal1:] if T is not None else None

        X_test = X[test_idx]
        Y_test = Y[test_idx]
        T_test = T[:, test_idx] if T is not None else None

        # Calculate mutual information for each feature
        def mutual_info_X(X, T):
            MI = []
            X_temp = np.array([hash(tuple(x)) for x in X])
            for i in range(T.shape[0]):
                T_temp = np.array([hash(tuple(x)) for x in T[i, :]])
                MI.append(mutual_info_score(X_temp, T_temp))
            return np.array(MI)
        def mutual_info_Y(X, T):
            MI = []
            for i in range(T.shape[0]):
                T_temp = np.array([hash(tuple(x)) for x in T[i,:]])
                MI.append(mutual_info_score(X, T_temp))
            return np.array(MI)

        acc_per_h_cal1 = np.array([mutual_info_Y(Y_cal1, T_cal1)])
        cost_per_h_cal1 = np.array([mutual_info_X(X_cal1, T_cal1)])
        acc_per_h_cal2 = np.array([mutual_info_Y(Y_cal2, T_cal2)])
        cost_per_h_cal2 = np.array([mutual_info_X(X_cal2, T_cal2)])
        acc_per_h_test = np.array([mutual_info_Y(Y_test, T_test)])
        cost_per_h_test = np.array([mutual_info_X(X_test, T_test)])
        ##########################################
        ############# Pareto Frontier ############
        ##########################################

        acc1 = acc_per_h_cal1.reshape(-1)
        cost1 = cost_per_h_cal1.reshape(-1)
        utilities = np.stack((-acc1, cost1), axis=1)

        is_efficient = is_pareto(utilities)
        all_ids = np.arange(acc1.shape[0])
        efficient_ids = all_ids[is_efficient]
        # Sort according to p-values

        if t==0:
            XT_pareto = cost_per_h_test[0, efficient_ids]
            YT_pareto = acc_per_h_test[0, efficient_ids]
            not_efficient = all_ids[~is_efficient]
            XT_not_pareto = cost_per_h_test[0, not_efficient]
            YT_not_pareto = acc_per_h_test[0, not_efficient]
            plt.figure(figsize=(12, 8))
            plt.rcParams['text.usetex'] = True
            plt.rcParams.update({'font.size': 32})
            plt.scatter(YT_pareto, XT_pareto, color='red', s=100, edgecolor='red',
                        marker='o')
            plt.xlabel(r'$\hat{I}_\lambda(T;Y)$')
            plt.ylabel(r'$\hat{I}_\lambda(X;T)$')
            plt.axvline(x=2.28, color='orange', linestyle='--', label=f'Target')
            #plt.title('Pareto Points')
            plt.xlim([2.26, 2.299])
            plt.ylim([8.4, 8.52])
            plt.tight_layout()
            plt.grid(True)
            plt.savefig('Pareto_front_2.pdf')
            plt.show()


        for a, alpha in enumerate(alphas):
            #print(f'{a}: alpha = {alpha}')

            ##########################################
            ######## Fixed Sequence Testing ##########
            ##########################################
            p_vals_YT = np.array(
                [IB_p_value_YT(Y_cal1, T_cal1[i, :], alpha) for i in range(T_cal1.shape[0])])
            p_vals = p_vals_YT
            efficent_sorted = efficient_ids[np.argsort(p_vals[is_efficient])]
            risk2 = acc_per_h_cal2[0, 0] - acc_per_h_cal2
            p_vals_YT = np.array([IB_p_value_YT(Y_cal2, T_cal2[i, :], alpha) for i in range(T_cal2.shape[0])])
            p_vals2 = p_vals_YT
            list_rejected = fixed_sequence_testing(efficent_sorted, p_vals2)
            ##########################################
            ######## Select ##########
            ##########################################
            score = [-cost_per_h_cal2[:, id_rej - 1] for id_rej in list_rejected]
            if len(score) > 0:
                id = score.index(max(score))
                id_max_score = list_rejected[id] - 1
                XT, YT = cost_per_h_test[0, id_max_score], acc_per_h_test[0, id_max_score]
                idC = 0 # Change to indicate how the id is chosen in the conventional method
                YTC = acc_per_h_test[0, idC]
                XTC = cost_per_h_test[0, idC]
                if XT != 0 and YT != 0:
                    XT_all += [XT]
                    YT_all += [YT]
                    XT_compare += [XTC]
                    YT_compare += [YTC]


    df = pd.DataFrame.from_dict(
        {"XT": np.array(XT_all), "YT": np.array(YT_all), "XTC": np.array(XT_compare), "YTC": np.array(YT_compare)})
    plots(df, alpha)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--res_folder", type=str, default='ag_pruning_results')
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--n_cal", type=int, default=1999)
    parser.add_argument("--n_cal1", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--alphas", type=str, default='0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2')

    args = parser.parse_args()

    # Modify the number of test and calibration data
    args.n_test = 10000
    args.n_cal = 4999
    args.n_cal1 = 2500
    args.n_trials = 33
    args.alphas = '2.28'

    main(args)
