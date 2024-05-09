from evaluator.performance import calculate_performance_metrics
from evaluator.evaluator import calculate_all_metrics
import pandas as pd

def ambiguity_rejection(max_rr, detail_factor, model, xt, all_data, file_path, experiment_id, dataset, rmse_accepted_perfect=[]):
    # Set variables
    reject_rates = []
    rmse_accepted = []
    rmse_rejected = []
    rmse_change_accepted = []
    optimal_reject_rate = None
    metrics_results_list = []
    min_rmse = float('inf')  # Set to positive infinity initially

    # No Rejection
    all_data['ite_reject'] = all_data['ite_pred']
    metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)
    reject_rates.append(metrics_result.get('Rejection Rate', None))
    rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
    rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
    rmse_original = metrics_result.get('RMSE Original', None)
    rmse_change_accepted.append(0)
    metrics_results_list.append(metrics_result)

    # Calculate the Ambiguity Scores

    y_lower = model.predict(xt, quantiles=[0.025])
    y_upper = model.predict(xt, quantiles=[0.975])

    y_lower2 = model.predict(xt, quantiles=[0.05])
    y_upper2 = model.predict(xt, quantiles=[0.95])

    # y_lower3 = model.predict(xt, quantiles=[0.10])
    # y_upper3 = model.predict(xt, quantiles=[0.90])

    # y_lower4 = model.predict(xt, quantiles=[0.15])
    # y_upper4 = model.predict(xt, quantiles=[0.85])

    # size_of_ci = ((y_upper - y_lower) + (y_upper2 - y_lower2) + (y_upper3 - y_lower3) + (y_upper4 - y_lower4)) /4 # confidence interval
    
    size_of_ci = ((y_upper - y_lower) + (y_upper2 - y_lower2)) /2 # confidence interval
    # all_data['Ambiguity Score'] = size_of_ci
    all_data['Ambiguity Score'] = all_data['size_of_ci']
    all_data = all_data.sort_values(by='Ambiguity Score', ascending=False).copy()
    all_data = all_data.reset_index(drop=True)

    all_data.to_csv(f'output/csv/{dataset}/{experiment_id}_{dataset}_ambiguity.csv', index=False)
    
    # Normalize the Ambiguity Score
    all_data['Ambiguity Score Normalized'] = (all_data['Ambiguity Score'] - all_data['Ambiguity Score'].min()) / (all_data['Ambiguity Score'].max() - all_data['Ambiguity Score'].min())

    # Calculate the performance metrics for each rejection rate
    for rr in range(1, max_rr * detail_factor):
        num_to_set = int(rr / (100.0 * detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data
        all_data['ite_reject'] = all_data['ite_pred'].astype(object)
        if num_to_set != 0:
            all_data.loc[:num_to_set - 1, 'ite_reject'] = 'R'

        metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)
        if metrics_result:
            reject_rates.append(metrics_result.get('Rejection Rate', None))
            rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
            rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
            rmse_original = metrics_result.get('RMSE Original', None)
            improvement = (metrics_result.get('RMSE Accepted', None) - rmse_original) / rmse_original * 100
            rmse_change_accepted.append(improvement)
            metrics_results_list.append(metrics_result)
        else:
            reject_rates.append(None)
            rmse_accepted.append(None)
            rmse_rejected.append(None)

    # export to csv
    data = {
        'Reject Rate': reject_rates,
        'RMSE Accepted': rmse_accepted,
        'RMSE Rejected': rmse_rejected,
        'Improvement': rmse_change_accepted
    }

    df = pd.DataFrame(data)
    df.to_csv(f'output/csv/{dataset}/{experiment_id}_{dataset}_ambiguity.csv', index=False)

    # Search for the reject rate with the lowest RMSE
    min_rmse = min(rmse_accepted)
    optimal_reject_rate = reject_rates[rmse_accepted.index(min_rmse)]

    filtered_data = all_data[all_data['Ambiguity Score'] != 0]
    plusstd = filtered_data['Ambiguity Score'].mean() + filtered_data['Ambiguity Score'].std()
    plusstd_normalized = filtered_data['Ambiguity Score Normalized'].mean() + filtered_data['Ambiguity Score'].std()

    heuristic_reject_rate = len(all_data[all_data['Ambiguity Score'] > plusstd]) / len(all_data)

    # Calculate the metric for the heuristic cutoff
    num_to_set = int(heuristic_reject_rate * len(all_data)) # example: 60/100 = 0.6 * length of the data
    all_data['ite_reject'] = all_data['ite_pred']
    all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
    all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'
    
    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, {}, append_metrics_results=False, print=False)
    metrics_dict['2/ Optimal RR (%)'] = round(optimal_reject_rate * 100, 4)
    metrics_dict['2/ Original RMSE'] = metrics_dict.get('RMSE Original', None)
    metrics_dict['2/ Minimum RMSE'] = round(min_rmse, 4)
    metrics_dict['2/ Change of RMSE (%)'] = (min_rmse - metrics_dict.get('RMSE Original', None)) / metrics_dict.get('RMSE Original', None) * 100
    metrics_dict['2/ Improvement of RMSE (%)'] = -((min_rmse - metrics_dict.get('RMSE Original', None)) / metrics_dict.get('RMSE Original', None)) * 100
    metrics_dict['2/ Mistake from Perfect'] = round(sum([perfect - actual for perfect, actual in zip(rmse_accepted_perfect, rmse_accepted)]), 4)

    # Return values
    return metrics_dict, reject_rates, heuristic_reject_rate, metrics_results_list
