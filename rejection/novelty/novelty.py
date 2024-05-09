from .helper import f, train_model
import matplotlib.pyplot as plt

from evaluator.performance import calculate_performance_metrics
from evaluator.evaluator import calculate_all_metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import pandas as pd

def novelty_rejection(type_nr, max_rr, detail_factor, model_name, x, all_data, file_path, experiment_id, dataset, rmse_accepted_perfect=[]):
    # 0 Preparation
    ## 0.1 Set up the lists to store the results
    reject_rates = []
    rmse_accepted = []
    rmse_rejected = []
    rmse_change_accepted = []
    metrics_results_list = []
    min_rmse = float('inf')
    optimal_model = None

    ## 0.2 No Rejection
    all_data['ite_reject'] = all_data['ite_pred']
    metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)
    reject_rates.append(metrics_result.get('Rejection Rate', None))
    rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
    rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
    rmse_change_accepted.append(0)
    metrics_results_list.append(metrics_result)

    # 1 Calculate the Novelty Score
    # 1.1 Calculate the Novelty Score for definition 1
    if type_nr == 1:
        all_data['Novelty Score'] = 0 # Amount of times rejected
        # Loop over all possible contamination (reject rate) values
        for contamination in range(int(1*detail_factor), int(max_rr*detail_factor)):
            contamination /= (100 * detail_factor) # max of 0.5

            if model_name == IsolationForest:
                model = train_model(x, IsolationForest, contamination=contamination, random_state=42) # lower contamination, less outliers
                all_data['ood'] = pd.Series(model.predict(x), name='ood')
            elif model_name == OneClassSVM:
                model = train_model(x, OneClassSVM, nu=contamination) # lower contamination, less outliers
                all_data['ood'] = pd.Series(model.predict(x), name='ood')
            elif model_name == LocalOutlierFactor:
                model = train_model(x, LocalOutlierFactor, contamination=contamination, novelty=True)
                all_data['ood'] = pd.Series(model.predict(x), name='ood')

            all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
            all_data['y_reject'] = all_data.apply(lambda row: True if row['ood'] == -1 else False, axis=1)

            set_rejected = all_data.copy()
            set_accepted = all_data.copy()
            set_rejected = set_rejected[set_rejected['y_reject'] == True]
            set_accepted = set_accepted[set_accepted['y_reject'] == False]

            all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)
            metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

            # Calculate metrics
            if metrics_result:
                reject_rates.append(metrics_result.get('Rejection Rate', None))
                current_rmse = metrics_result.get('RMSE Accepted', None)
                rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
                rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
                improvement = ( metrics_result.get('RMSE Accepted', None) - metrics_result.get('RMSE Original', None) ) / metrics_result.get('RMSE Original', None) * 100
                metrics_results_list.append(metrics_result)
                rmse_change_accepted.append(improvement)
            else:
                reject_rates.append(None)
                rmse_accepted.append(None)
                rmse_rejected.append(None)

            # Update minimum RMSE and optimal model if needed
            if current_rmse < min_rmse:
                min_rmse = current_rmse
                optimal_model = model
        
        # Calculate the heuristic rejection rate
        all_data['y_reject'] = all_data.apply(lambda row: True if row['Novelty Score'] > 0 else False, axis=1)
        all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)
        heuristic_reject_rate = len(all_data[all_data['Novelty Score'] > 0]) / len(all_data)

    # 1.1 Calculate the Novelty Score for definition 2 and 3
    if type_nr == 2 or type_nr == 3:
        # split the data
        t_data = all_data[all_data['treatment'] == 1].copy()
        ut_data = all_data[all_data['treatment'] == 0].copy()
        t_x = all_data[all_data['treatment'] == 1].filter(regex='^feature_').copy()
        ut_x = all_data[all_data['treatment'] == 1].filter(regex='^feature_').copy()
        # t_x = all_data[all_data['treatment'] == 1].copy()
        # ut_x = x[all_data['treatment'] == 0].copy()
        
        t_data['Novelty Score'] = 0 # Amount of times rejected
        ut_data['Novelty Score'] = 0
        all_data['Novelty Score'] = 0

        for contamination in range(int(1 * detail_factor), int(49 * detail_factor)):
            # Calculate the Novelty Score
            amount_of_times_rejected_new = f(type_nr, contamination, t_x, ut_x, t_data, ut_data, detail_factor, model_name, all_data)
            all_data['Novelty Score'] += amount_of_times_rejected_new
            all_data['Novelty Score'].fillna(0, inplace=True)
            all_data['Novelty Score'] = all_data['Novelty Score'].astype(int)

        # Loop over all possible rejection rates
        all_data = all_data.sort_values(by='Novelty Score', ascending=False).copy()
        all_data = all_data.reset_index(drop=True)
        for rr in range(1, max_rr*detail_factor):
            num_to_set = int(rr / (100.0*detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data

            all_data['ite_reject'] = all_data['ite_pred']
            all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
            if num_to_set != 0:
                all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

            metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

            if metrics_result:
                reject_rates.append(metrics_result.get('Rejection Rate', None))
                rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
                rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
                improvement = ( metrics_result.get('RMSE Accepted', None) - metrics_result.get('RMSE Original', None) ) / metrics_result.get('RMSE Original', None) * 100
                rmse_change_accepted.append(improvement)
                metrics_results_list.append(metrics_result)
            else:
                reject_rates.append(None)
                rmse_accepted.append(None)
                rmse_rejected.append(None)
        
        # Calculate the heuristic rejection rate
        heuristic_reject_rate = len(all_data[all_data['Novelty Score'] > 0]) / len(all_data)

    # Find the lowest RMSE (optimal model)
    min_rmse = min(rmse_accepted)  # Find the minimum
    min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
    optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

    # export to csv
    data = {
        'Reject Rate': reject_rates,
        'RMSE Accepted': rmse_accepted,
        'RMSE Rejected': rmse_rejected,
        'Improvement': rmse_change_accepted
    }

    df = pd.DataFrame(data)
    df.to_csv(f'output/csv/{dataset}/{experiment_id}_{dataset}.csv', index=False)

    # Calculate the metrics for heuristic rejection rate
    num_to_set = int(heuristic_reject_rate*100 / (100.0*detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data

    all_data['ite_reject'] = all_data['ite_pred']
    all_data['rejected'] = False
    all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
    if num_to_set != 0:
        all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'
        all_data.loc[:num_to_set -1, 'rejected'] = True

    all_data.to_csv(f'output/data/{dataset}/{experiment_id}_all_data.csv', index=False)

    # Create the histogram
    all_data['R']= all_data['rejected'].apply(lambda x: 1 if x == True else 0)
    all_data['T']= all_data['rejected'].apply(lambda x: 1 if x == True else 1)
    if dataset == "TWINSC":
        plt.hist(all_data['ite_sign'], bins=(-1,0,1,2), align='left', rwidth=0.8, weights=all_data['T'], color='grey')
        plt.hist(all_data['ite_sign'], bins=(-1,0,1,2), align='left', rwidth=0.8, weights=all_data['R'], color='red')
        ## Set labels and title
        plt.xlabel('Sign of TE')
        plt.ylabel('Rejected samples in red')
        plt.title('Histogram of Count of Rejected Observations by Decile')
        ## Adjust layout and display the plot
        plt.xticks((-1,0,1))
    else:
        plt.hist(all_data['decile'], bins=range(11), align='left', rwidth=0.8, weights=all_data['T'], color='grey')
        plt.hist(all_data['decile'], bins=range(11), align='left', rwidth=0.8, weights=all_data['R'], color='red')
        ## Set labels and title
        plt.xlabel('Decile of TE')
        plt.ylabel('Rejected samples in red')
        plt.title('Histogram of Count of Rejected Observations by Decile')
        ## Adjust layout and display the plot
        plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(f"output/data/{dataset}/{experiment_id}_histogram.png")
    plt.close()

    # Create the stack bar (se)
    if dataset == "TWINSC":
        se_r_sum = all_data.loc[all_data['rejected'] == True].groupby('category')['se'].sum()
        se_not_r_sum = all_data.loc[all_data['rejected'] == False].groupby('category')['se'].sum()
        # Get the index values for the deciles
        category_index = se_not_r_sum.index
        se_r_sum = se_r_sum.reindex(category_index, fill_value=0)

        plt.bar(category_index, se_not_r_sum, label='SE for accepted observations', color='green')
        plt.bar(category_index, se_r_sum, bottom=se_not_r_sum, label='SE for rejected observations', color='red')

        # Set labels and title
        plt.xlabel('Decile')
        plt.ylabel('Sum of Squared Error of TE')
        plt.title('Sum of Squared Error of TE by Category')
        # plt.tight_layout()
        plt.savefig(f"output/data/{dataset}/{experiment_id}_stacked_bar.png")
        plt.close()

        se_sum = all_data.groupby('category')['se'].sum()
        # Get the index values for the deciles
        category_index = se_sum.index

        plt.bar(category_index, se_sum, label='SE for all observations', color='grey')

        # Set labels and title
        plt.xlabel('Category')
        plt.ylabel('Sum of Squared Error of TE')
        plt.title('Sum of Squared Error of TE by Category')
        # plt.tight_layout()
        plt.savefig(f"output/data/{dataset}/{experiment_id}_Total_stacked_bar.png")
        plt.close()

    else:
        se_r_sum = all_data.loc[all_data['rejected'] == True].groupby('decile')['se'].sum()
        se_not_r_sum = all_data.loc[all_data['rejected'] == False].groupby('decile')['se'].sum()
        # Get the index values for the deciles
        decile_index = se_not_r_sum.index
        se_r_sum = se_r_sum.reindex(decile_index, fill_value=0)

        plt.bar(decile_index, se_not_r_sum, label='SE for accepted observations', color='green')
        plt.bar(decile_index, se_r_sum, bottom=se_not_r_sum, label='SE for rejected observations', color='red')

        # Set labels and title
        plt.xlabel('Decile')
        plt.ylabel('Sum of Squared Error of TE')
        plt.title('Sum of Squared Error of TE by Decile')
        # plt.tight_layout()
        plt.savefig(f"output/data/{dataset}/{experiment_id}_stacked_bar.png")
        plt.close()

        se_sum = all_data.groupby('decile')['se'].sum()
        # Get the index values for the deciles
        decile_index = se_sum.index

        plt.bar(decile_index, se_sum, label='SE for all observations', color='grey')

        # Set labels and title
        plt.xlabel('Decile')
        plt.ylabel('Sum of Squared Error of TE')
        plt.title('Sum of Squared Error of TE by Decile')
        # plt.tight_layout()
        plt.savefig(f"output/data/{dataset}/{experiment_id}_Total_stacked_bar.png")
        plt.close()


    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, {}, append_metrics_results=False, print=False)

    metrics_dict['2/ Optimal RR (%)'] = round(optimal_reject_rate, 4)*100

    original_rmse = metrics_dict.get('RMSE Original', None)
    metrics_dict['2/ Original RMSE ()'] = original_rmse
    metrics_dict['2/ Minimum RMSE'] = round(min_rmse, 4)

    metrics_dict['2/ Change of RMSE (%)'] = (min_rmse - original_rmse) / original_rmse * 100
    metrics_dict['2/ Improvement of RMSE (%)'] = -((min_rmse - original_rmse) / original_rmse) * 100

    mistake_from_perfect_column = [perfect - actual for perfect, actual in zip(rmse_accepted_perfect, rmse_accepted)]
    mistake_from_perfect = sum(mistake_from_perfect_column)
    metrics_dict['2/ Mistake from Perfect'] = round(mistake_from_perfect, 4)
    
    # Return the results
    return metrics_dict, reject_rates, heuristic_reject_rate, metrics_results_list