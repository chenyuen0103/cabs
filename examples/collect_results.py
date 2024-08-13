import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']

# Directories
results_dir = 'results'
plot_dir = 'plots'
avg_dir = 'averaged'

# Columns for which we want to calculate the standard errors


col_name_map = {'Step': 'step',
                'Loss': 'train_loss',
                'Train_loss': 'train_loss',
                'Train Accuracy':'train_acc',
                'Test Accuracy':'test_acc',
                'batch_size':'batch_sizes',
                'Batch Size': 'batch_sizes',
                'Gradient Diversity': 'grad_div',
                'Time': 'time'
                }

avg_columns = col_name_map.values()

# Regular expression patterns to extract parameters from filenames
pattern_cabs = re.compile(
    r'(?P<method>.+)_(?P<dataset>.+)_s(?P<seed>\d+)\.csv')
pattern_ours = re.compile(
    r'(?P<method>our)_(?P<dataset>cifar\d+)_delta(?P<delta>\d+\.\d+)_s(?P<seed>\d+)\.csv')
def create_directories():
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(avg_dir):
        os.makedirs(avg_dir)

def parse_filename(filename):
    match_cabs = pattern_cabs.match(filename)
    match_ours = pattern_ours.match(filename)

    if 'cabs' in filename and match_cabs:
        params = match_cabs.groupdict()
    elif 'our' in filename and match_ours:
        params = match_ours.groupdict()
    else:
        print(f"{filename} does not match any pattern.")
        return None, None

    method = params['method']
    dataset = params['dataset']
    seed = int(params['seed'])
    delta = params.get('delta')
    if delta is not None:
        delta = float(delta)

    key = (method, dataset, delta)

    return key, seed

def read_results():
    results = {}
    for filename in os.listdir(results_dir):
        key, seed = parse_filename(filename)
        if key is not None:
            if seed > 5:
                continue
            path = os.path.join(results_dir, filename)
            try:
                df = pd.read_csv(path)
                # If the last step was logged twice, delete the first occurrence

                # renmae columns
                df.rename(columns=col_name_map, inplace=True)
                if df['step'].duplicated().any():
                    df = df[~df['step'].duplicated(keep='last')]
                if len(df) < 100:
                    print(f"{filename} has only {len(df)} rows.")
                    continue
                if key not in results:
                    results[key] = []
                results[key].append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return results


def plot_results(results, dataset = None, lr_plot = 0.1, save = False):
    name_map = {'cabs': 'CABS', 'our': 'DiveBatch'}
    label_map = {'train_loss': 'Loss', 'train_acc': 'Accuracy', 'test_acc': 'Accuracy','batch_sizes': 'Batch Size'}
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    method_colors = {
        'cabs': CB_color_cycle[2],
        'our': CB_color_cycle[3]
    }


    # plot results for len(results[key]) >= 5
    results = {key: dfs for key, dfs in results.items() if len(dfs) >= 5}
    # For "ours" method, filter results for highest validation accuracy

    if dataset is None:
        dataset_list = ['cifar10', 'cifar100']
    else:
        dataset_list = [dataset]

    for dataset in dataset_list:
        data_results = {key: dfs for key, dfs in results.items() if key[1] == dataset}


        for metric in label_map.keys():
            if save:
                plt.style.use(['science','ieee'])
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)
                rcParams['text.usetex'] = True
                rcParams['font.family'] = 'serif'
                rcParams['font.serif'] = ['Computer Modern Roman']

            # plt.figure(figsize=(10, 6))
            # Set the color cycle
            # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)
            handles, labels = [], []
            for key, dfs in data_results.items():
                if key[1] == dataset:  # Filter by method 'adabatch' or 'fixed' and dataset
                    method, _,delta = key
                    combined_results = pd.concat(dfs).groupby(level=0)

                    linestyle = ':' if (method == 'fixed') else '-'
                    linewidth = 2 if (method == 'our') else 1
                    zorder = 1 if (method == 'our') else 2
                    color = method_colors.get(method, CB_color_cycle[0])
                    if metric not in combined_results.first():
                        print(f"Skipping {key} due to missing {metric} column.")
                        continue
                    mean_results = combined_results[metric].mean()
                    mean_results = mean_results.ffill()
                    batch_size_avg = combined_results['batch_sizes'].mean()
                    # Adjust x-axis for test accuracy logging interval
                    if metric == 'test_acc':
                        x_values = mean_results.index * 100 if metric == 'test_acc' else mean_results.index

                    if not save:
                        label = f'{name_map[method]}'
                        if delta is not None:
                            label = label + r'($\delta$='+ f'{delta})'

                    else:
                        label = f'{name_map[method]}'
                        # compute the end accuracy and standard error
                        end_accuracy = mean_results.iloc[-1]
                        std_error = combined_results[metric].sem().iloc[-1]
                        # label += f' ({end_accuracy:.2f} $\pm$ {std_error:.2f})'
                    # plt.plot(mean_results.index, mean_results, label=label, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
                    if save:
                        line, = plt.plot(combined_results.first()['step'], mean_results, label=label, linewidth=linewidth, linestyle=linestyle, zorder=zorder, color=color)
                        # line, = plt.plot(mean_results.index, mean_results, label=label)
                        handles.append(line)
                        labels.append(label)
                    else:
                        plt.plot(combined_results.first()['step'], mean_results, label=label, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
            plt.xlabel('Step')
            # plt.ylabel(metric.replace('_', ' ').title())
            plt.ylabel(label_map[metric])
            # if 'loss' in metric:
            #     plt.yscale('log')
            if not save:
                plt.title(f'{dataset.upper()} (lr: {lr_plot}) - {metric.replace("_", " ").title()}')
            else:
                plt.title(f'{dataset.upper()}')
            # Create a custom legend order
            # Sort legend entries to have 'Fixed' with smaller batch size first, followed by 'cabs' and then 'DiveBatch'

            if save:
                # fixed_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'SGD' in l]
                # cabs_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'CABS' in l]
                proposed_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'DiveBatch' in l]

                # fixed_handles_labels = sorted(fixed_handles_labels, key=lambda x: int(re.search(r'\((\d+)\)', x[1]).group(1)))
                # sorted_handles_labels = cabs_handles_labels + proposed_handles_labels
                sorted_handles_labels = proposed_handles_labels
                handles, labels = zip(*sorted_handles_labels)

            # Sort legend entries
            # sorted_handles_labels = sorted(zip(lines, labels), key=lambda x: (
            # 'Fixed' in x[1],
            # int(x[1].split('(')[1].split(')')[0])))
            # handles, labels = zip(*sorted_handles_labels)

            # plt.legend(sorted_lines, sorted_labels)

            # plt.grid(axis='y')
            # plt.show()
            if save:
                plt.grid(linestyle='dotted')
                plt.legend(handles, labels, loc='best', fontsize=10)
                plt.savefig(f'{plot_dir}/{dataset}_{metric}_lr{lr_plot}.pdf', format='pdf')
            else:
                plt.legend()
                plt.show()
            plt.close()




def generate_latex_table_for_epochs(results, lr_plot):
    name_map = {'fixed': 'SGD', 'cabs': 'CABS', 'our': '\\ourmethod'}
    table_content = {'cifar10': [], 'cifar100': []}

    for dataset in table_content.keys():
        filtered_results = {key: dfs for key, dfs in results.items() if key[1] == dataset and key[4] == lr_plot}

        # Filter for the top "ours" configuration
        ours_results = {key: dfs for key, dfs in filtered_results.items() if key[0] == 'our'}
        if ours_results:
            if dataset == 'cifar10' and lr_plot == 0.01:
                top_ours_key, top_ours_dfs = ('our', 'cifar10', 'resnet', 128, 0.01, 0.005, '20'), ours_results[
                    ('our', 'cifar10', 'resnet', 128, 0.01, 0.005, '20')]
            elif dataset == 'cifar10' and lr_plot == 0.1:
                top_ours_key, top_ours_dfs = ('our', 'cifar10', 'resnet', 128, 0.1, 0.005, '20'), ours_results[
                    ('our', 'cifar10', 'resnet', 128, 0.1, 0.005, '20')]
            elif dataset == 'cifar100' and lr_plot == 0.1:
                top_ours_key, top_ours_dfs = ('our', 'cifar100', 'resnet', 128, 0.1, 0.01, '20'), ours_results[
                    ('our', 'cifar100', 'resnet', 128, 0.1, 0.005, '20')]
            elif dataset == 'cifar100' and lr_plot == 0.01:
                top_ours_key, top_ours_dfs = ('our', 'cifar100', 'resnet', 128, 0.01, 0.0005, '20'), ours_results[
                    ('our', 'cifar100', 'resnet', 128, 0.01, 0.0005, '20')]

            filtered_results = {key: dfs for key, dfs in filtered_results.items() if key[0] != 'our'}
            filtered_results[top_ours_key] = top_ours_dfs

        for key, dfs in filtered_results.items():
            method, _, model, batch_size, lr, delta, rf = key

            epoch_thresholds_for_trials = []

            for df in dfs:
                # Calculate end accuracy for each trial
                end_accuracy = df['test_acc'].iloc[-1]

                # Calculate the +- 1% range for this trial
                accuracy_threshold_low = end_accuracy * 0.99
                accuracy_threshold_high = end_accuracy * 1.01

                # Find the epoch where accuracy falls within +- 1% of the end accuracy
                epoch_within_threshold = None
                for epoch in df.index:
                    current_accuracy = df['test_acc'].loc[epoch]
                    if accuracy_threshold_low <= current_accuracy <= accuracy_threshold_high:
                        epoch_within_threshold = epoch
                        break  # Exit the loop once the threshold is met

                if epoch_within_threshold is not None:
                    epoch_thresholds_for_trials.append(epoch_within_threshold)

            # Compute the average of epochs and standard error across all trials
            if epoch_thresholds_for_trials:
                avg_epoch_within_threshold = np.mean(epoch_thresholds_for_trials)
                sem_epoch_within_threshold = np.std(epoch_thresholds_for_trials, ddof=1) / np.sqrt(
                    len(epoch_thresholds_for_trials))
            else:
                avg_epoch_within_threshold = None
                sem_epoch_within_threshold = None

            label = f'{name_map[method]} - {batch_size}' if method == 'fixed' else f'{name_map[method]}'
            table_content[dataset].append((label, avg_epoch_within_threshold, sem_epoch_within_threshold, method))

        # Sort table content for 'fixed' method by batch size
        table_content[dataset].sort(
            key=lambda x: (x[3] == 'fixed', int(re.search(r'- (\d+)', x[0]).group(1)) if 'SGD' in x[0] else 0))

    # Generate LaTeX table
    latex_table = "\\begin{table}[h!]\n\\centering\n"
    latex_table += "\\caption{Average number of epochs to reach within ±1\\% of the final test accuracy for ResNet-20 on \\textsc{CiFar}-10 and \\textsc{CiFar}-100. The values in parentheses represent the standard error of the mean.}\n"
    latex_table += "\\begin{adjustbox}{max width=\\textwidth}\n"
    latex_table += "\\begin{tabular}{llc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Dataset} & \\textbf{Algorithm} & \\textbf{Avg. Epochs} \\\\\n"
    latex_table += "\\midrule\n"

    for dataset, content in table_content.items():
        latex_table += f"\\multirow{{{len(content)}}}{{*}}{{{dataset.upper()}}} "

        # Sort by Fixed - 128, Fixed - 2048, CABS, \ourmethod
        fixed_128 = [entry for entry in content if 'SGD - 128' in entry[0]]
        fixed_2048 = [entry for entry in content if 'SGD - 2048' in entry[0]]
        cabs = [entry for entry in content if 'CABS' in entry[0]]
        divebatch = [entry for entry in content if '\\ourmethod' in entry[0]]
        sorted_content = fixed_128 + fixed_2048 + cabs + divebatch

        for i, (label, avg_epoch_within_threshold, sem_epoch_within_threshold, method) in enumerate(sorted_content):
            latex_table += f"& {label} & "
            if avg_epoch_within_threshold is not None:
                latex_table += f"{avg_epoch_within_threshold:.1f} \\\\\n"
            else:
                latex_table += "N/A \\\\\n"
        latex_table += "\\midrule\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{adjustbox}\n"
    latex_table += "\\end{table}\n"

    return latex_table



def find_avg_epochs_within_threshold(results, lr_plot):
    name_map = {'fixed': 'SGD', 'cabs': 'CABS', 'our': '\ourmethod'}
    table_content = {'cifar10': [], 'cifar100': []}
    epoch_stats = {}

    for dataset in table_content.keys():
        filtered_results = {key: dfs for key, dfs in results.items() if key[1] == dataset and key[4] == lr_plot}

        # Filter for the top "ours" configuration
        ours_results = {key: dfs for key, dfs in filtered_results.items() if key[0] == 'our'}
        if ours_results:
            if dataset == 'cifar10' and lr_plot == 0.01:
                top_ours_key, top_ours_dfs = ('our', 'cifar10', 'resnet', 128, 0.01, 0.005, '20'), ours_results[
                    ('our', 'cifar10', 'resnet', 128, 0.01, 0.005, '20')]
            elif dataset == 'cifar10' and lr_plot == 0.1:
                top_ours_key, top_ours_dfs = ('our', 'cifar10', 'resnet', 128, 0.1, 0.005, '20'), ours_results[
                    ('our', 'cifar10', 'resnet', 128, 0.1, 0.005, '20')]
            elif dataset == 'cifar100' and lr_plot == 0.1:
                top_ours_key, top_ours_dfs = ('our', 'cifar100', 'resnet', 128, 0.1, 0.01, '20'), ours_results[
                    ('our', 'cifar100', 'resnet', 128, 0.1, 0.005, '20')]
            elif dataset == 'cifar100' and lr_plot == 0.01:
                top_ours_key, top_ours_dfs = ('our', 'cifar100', 'resnet', 128, 0.01, 0.0005, '20'), ours_results[
                    ('our', 'cifar100', 'resnet', 128, 0.01, 0.0005, '20')]

            filtered_results = {key: dfs for key, dfs in filtered_results.items() if key[0] != 'our'}
            filtered_results[top_ours_key] = top_ours_dfs

        for key, dfs in filtered_results.items():

            epoch_thresholds_for_trials = []

            for df in dfs:
                # Calculate end accuracy for each trial
                end_accuracy = df['test_acc'].iloc[-1]

                # Calculate the +- 1% range for this trial
                accuracy_threshold_low = end_accuracy * 0.99
                accuracy_threshold_high = end_accuracy * 1.01

                # Find the epoch where accuracy falls within +- 1% of the end accuracy
                epoch_within_threshold = None
                for epoch in df.index:
                    current_accuracy = df['test_acc'].loc[epoch]
                    if accuracy_threshold_low <= current_accuracy <= accuracy_threshold_high:
                        epoch_within_threshold = epoch
                        break  # Exit the loop once the threshold is met

                if epoch_within_threshold is not None:
                    epoch_thresholds_for_trials.append(epoch_within_threshold)

            # Compute the average of epochs and standard error across all trials
            if epoch_thresholds_for_trials:
                avg_epoch_within_threshold = np.mean(epoch_thresholds_for_trials)
                sem_epoch_within_threshold = np.std(epoch_thresholds_for_trials, ddof=1) / np.sqrt(
                    len(epoch_thresholds_for_trials))
            else:
                avg_epoch_within_threshold = None
                sem_epoch_within_threshold = None

            # Store the average epoch and standard error result in the dictionary
            epoch_stats[key] = {
                'avg_epoch': avg_epoch_within_threshold,
                'sem_epoch': sem_epoch_within_threshold
            }

    return epoch_stats


def main():
    create_directories()
    results = read_results()
    plot_results(results, dataset= 'cifar100', save = False)
    plot_results(results, dataset= 'cifar10', save = False)
    # latex_table = generate_latex_table(results, 0.1)
    # threshold = find_avg_epochs_within_threshold(results, 0.1)
    # print(threshold)
    # table = generate_latex_table_for_epochs(results, 0.1)
    # print(table)
    # print(latex_table)

if __name__ == "__main__":
    main()
