import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_summary(reject_rates_list, rmse_rank_accepted_list, experiment_ids_list, dataset, folder_path, plot_title, file_name):
    plt.figure(figsize=(10, 6))

    for i in range(1,10):
        plt.plot(reject_rates_list[i], rmse_rank_accepted_list[i], label=f"Experiment {experiment_ids_list[i]}")

    plt.xlabel('Reject Rate')
    plt.ylabel(f'{file_name}')
    plt.title(f'{plot_title} for {dataset}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{folder_path}graph/{dataset}_All_{file_name}.pdf")
    plt.close()
    plt.cla()

def plot_canvas(reject_rates_list, rmse_rank_accepted_list, experiment_ids_list, dataset, folder_path, plot_title, file_name):
    plt.figure(figsize=(15, 15))  # Increase the figure size for a 3x3 grid

    # Create a 3x3 grid of subplots
    for i in range(1, 10):
        plt.subplot(3, 3, i)

        # Plot the corresponding graph
        plt.plot(reject_rates_list[i], rmse_rank_accepted_list[i], label=f"Experiment {experiment_ids_list[i]}")
        plt.xlabel('Reject Rate')
        plt.ylabel(f'{file_name}')
        plt.title(f'Experiment {experiment_ids_list[i]}')
        plt.legend()
        plt.grid(True)

    plt.suptitle(f'{plot_title} for {dataset}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}graph/{dataset}_All_{file_name}.pdf")
    plt.close()
    plt.cla()

def canvas_change(reject_rates_list, metric_name, metric_name2, metric3, metric4,  metric5, metric_list, experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, xlabel, ylabel, folder, y_min, y_max, title):
    plt.figure(figsize=(15, 15))  # Increase the figure size for a 3x3 grid
    
    # Create a 3x3 grid of subplots
    for i in range(1, 10):
        plt.subplot(3, 3, i)

        # Plot the corresponding graph

        plt.plot([rate * 100 for rate in reject_rates_list[i]], [result.get(metric_name, None) for result in metric_list[i]], color="darkgray", label=f"{metric_name}")

        plt.plot([rate * 100 for rate in reject_rates_list[i]], [result.get(metric_name2, None) for result in metric_list[i]], color="gray", label=f"{metric_name2}")

        plt.plot([rate * 100 for rate in reject_rates_list[i]], [result.get(metric_name2, None) for result in metric_list[i]], color="dimgray", label=f"{metric3}")

        plt.plot([rate * 100 for rate in reject_rates_list[i]], [result.get(metric_name2, None) for result in metric_list[i]], color="black", label=f"{metric4}")

        plt.plot([rate * 100 for rate in reject_rates_list[i]], [result.get(metric_name2, None) for result in metric_list[i]], color="black", label=f"{metric5}")

        plt.xlim(0, 15)  # Set x-axis range from 0 to 6
        plt.ylim(y_min, y_max)  # Set x-axis range from 0 to 6
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        if heuristic_cutoff_list[i]*100 < 15:
            plt.axvline(x=heuristic_cutoff_list[i]*100, color='green', linestyle='-', linewidth=1, label='Heuristic Optimal RR') # this line is the heuristical cut-off point
            # Add text label for the vertical line
            plt.text(heuristic_cutoff_list[i]*100+0.25, -4, 'Heuristic Optimal RR', rotation=90, color='green', verticalalignment='center')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.title(f'Experiment {experiment_ids_list[i]}')
        plt.legend()
        # plt.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}overleaf/{folder}/{dataset}_All.pdf")
    plt.close()
    plt.cla()
    

def canvas_change_loop(reject_rates_list, metric_list, metric_name, experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, xlabel, ylabel, folder, y_min, y_max, title, datasets):
    plt.figure(figsize=(15, 15))  # Increase the figure size for a 3x3 grid
    
    # Create a 3x3 grid of subplots
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.xlim(0, 15)  # Set x-axis range from 0 to 6
        plt.ylim(y_min, y_max)  # Set x-axis range from 0 to 6
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        for dataset in datasets:
            # Plot the graph for the green color
            if dataset=="TWINSC":
                color = "green"
            else:
                color = "blue"
            plt.plot([rate * 100 for rate in reject_rates_list[dataset][i]], [result.get(metric_name, None) for result in metric_list[dataset][i]], color=color, label=f"{dataset}")

            # Check if heuristic cutoff is less than 15
            if heuristic_cutoff_list[dataset][i] * 100 < 15:
                plt.axvline(x=heuristic_cutoff_list[dataset][i]*100, color=color, linestyle=':', linewidth=1)
                plt.text(heuristic_cutoff_list[dataset][i]*100 + 0.25, y_min+(y_max-y_min)*0.25, 'Heuristic Optimal RR', rotation=90, color=color, verticalalignment='center')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.title(f'Experiment {experiment_ids_list[dataset][i]}')
        # plt.grid(True)
        plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}overleaf/{folder}/All_All.pdf")
    plt.close()
    plt.cla()

    plt.figure(figsize=(15, 15))  # Increase the figure size for a 2x1 grid

    # Create a 2x1 grid of subplots
    x = 0
    for i in [5,6,7,10]:
        x += 1
        plt.subplot(2, 2, x)  # Configure subplot as 2x1
    
        plt.ylim(y_min, y_max)  # Set x-axis range from 0 to 6
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        for dataset in datasets:
            # Plot the graph for the green color
            if dataset=="TWINSC":
                color = "green"
            else:
                color = "blue"
            plt.plot([rate * 100 for rate in reject_rates_list[dataset][i]], [result.get(metric_name, None) for result in metric_list[dataset][i]], color=color, label=f"{dataset}")

            # Check if heuristic cutoff is less than 15
            if heuristic_cutoff_list[dataset][i] * 100 < 15:
                plt.axvline(x=heuristic_cutoff_list[dataset][i]*100, color=color, linestyle=':', linewidth=1)
                plt.text(heuristic_cutoff_list[dataset][i]*100 + 0.25, -4, 'Heuristic Optimal RR', rotation=90, color=color, verticalalignment='center')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.title(f'Experiment {experiment_ids_list[dataset][i]}')
        # plt.grid(True)
        plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}overleaf/{folder}/Nov_vs_Amb.pdf")
    plt.close()
    plt.cla()


# Confidence Interval for Ambiguity Rejection
def confidence_interval(xt, forest_model):
    y_lower = forest_model.predict(xt, quantiles=[0.025])
    y_upper = forest_model.predict(xt, quantiles=[0.975])

    y_lower2 = forest_model.predict(xt, quantiles=[0.05])
    y_upper2 = forest_model.predict(xt, quantiles=[0.95])

    y_lower3 = forest_model.predict(xt, quantiles=[0.10])
    y_upper3 = forest_model.predict(xt, quantiles=[0.90])

    y_lower4 = forest_model.predict(xt, quantiles=[0.15])
    y_upper4 = forest_model.predict(xt, quantiles=[0.85])

    size_of_ci = ((y_upper - y_lower) + (y_upper2 - y_lower2) + (y_upper3 - y_lower3) + (y_upper4 - y_lower4)) /4 # confidence interval

    return size_of_ci