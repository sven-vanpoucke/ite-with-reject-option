# Chapter 0: Imports

## 0.0 General Packages
import pandas as pd
from tabulate import tabulate
import time

## 0.1 Packages for the data retrieval and preprocessing
from helper.output import helper_output_loop
from data.twins import load_data_twin, transform_data_twin
from data.processing import preprocessing_split_t_c_data, merge_test_train
from data.ihdp import transform_data_ihdp, preprocessing_transform_data_ihdp

## 0.2 Treatment Effect Estimation Model
from ite.model import t_model
from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from ite.predictor import predictor_train_predictions, predictor_test_predictions, predictor_ite_predictions

## 0.3 Evaluation
from evaluator.cost import categorize, categorize_decision
from evaluator.performance import calculate_performance_metrics
from evaluator.evaluator import calculate_all_metrics

## 0.4 Graphs
from rejection.helper import onelinegraph, twolinegraph
from helper.graphs import plot_summary, plot_canvas, canvas_change

# 0.5 Rejection
from rejection.ambiguity.ambiguity import ambiguity_rejection
from rejection.novelty.novelty import novelty_rejection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# 0.6: Evaluation
from helper.graphs import canvas_change_loop
import pandas as pd

# Chapter 1: Data retrieval and preprocessing
## 1.1 Set the parameters
datasets = ["TWINSC", "IHDP"] # Choose out of TWINS or TWINSC (if you want TWINS to be treated as continuous instead of classification) or LALONDE or IHDP
psm = False
detail_factor = 1 # This number can be set to 10 in case extra dettail in de graphs is wished. Default value is 1.
max_rr = 15 # This number indicates the maximal reject rate (%). This number can be between 1 and 49.
x_scaling = False # Set to True in case it's wished to scale the x variables. Otherwise, set to False.

## 1.2: Do not change these parameters
folder_path = 'output/'
text_folder_path = 'output/text/'
metrics_results = {}
experiment_names = {}
timestamp, file_name, file_path = helper_output_loop(folder_path=text_folder_path)
all_data_list = []
xt_list = []
x_list = []
train_forest_model_list = []
start_time = time.time()

# Metrics for the rejection
reject_rates_list = {}
rmse_rank_accepted_list = {}
rmse_rank_weighted_accepted_list = {}
sign_error_accepted_list = {}
signerror_weighted_accepted_list = {}
experiment_ids_list = {}
rmse_accepted_list = {}
rmse_change_accepted_list = {}
heuristic_cutoff_list = {}
sign_error_change_accepted_list = {}
rmse_rank_change_accepted_list = {}
rmse_rank_weighted_change_accepted_list = {}
metrics_results_list_global = {}

## 1.3: Do not change these parameters
for dataset in datasets: 
    if dataset == "TWINS" or dataset == "TWINSC":
        # Set the model that will be used in the T-Learner for the treatment effect
        model_class = LogisticRegression 
        model_params = {"max_iter": 10000, "solver": "saga", "random_state": 42}

        # Load the data
        train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = load_data_twin()

        # Transform the data
        train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)
        train_t.to_csv(f'output/csv2/{dataset}/0_train_t.csv', index=False)
        test_t.to_csv(f'output/csv2/{dataset}/0_test_t.csv', index=False)

    elif dataset == "IHDP":
        # Set the model that will be used in the T-Learner for the treatment effect
        model_class = LinearRegression
        model_params = {"fit_intercept": True}

        # Load the data
        train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y = transform_data_ihdp()
        # Transform the data
        train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y = preprocessing_transform_data_ihdp(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)

    # Transform the data
    train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)
    test_ite = pd.DataFrame({'ite': test_potential_y["y_t1"] - test_potential_y["y_t0"]})
    train_ite = pd.DataFrame({'ite': train_potential_y["y_t1"] - train_potential_y["y_t0"]})
    treated_x, treated_y, control_x, control_y, x, t, xt, train_xt, test_xt, y, ite, potential_y = merge_test_train(train_treated_x, train_treated_y, train_control_x, train_control_y, test_treated_x, test_treated_y, test_control_x, test_control_y, train_x, train_t, train_y, test_x, test_t, test_y, train_ite, test_ite, train_potential_y, test_potential_y, x_scaling)
    
    x.to_csv(f'output/csv2/{dataset}/0_ x.csv', index=False)
    t.to_csv(f'output/csv2/{dataset}/0_t.csv', index=False)

    # Chapter 2: Treatment Effect Estimation Model
    ## 2.1: T-Learner
    ### 2.1.1: Train models on the train_set
    train_treated_model, train_control_model = t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params)
    #### 2.1.1.1: Predictions of train_set
    train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(train_treated_model, train_control_model, train_treated_x, train_control_x)
    train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(train_treated_model, train_control_model, train_x)
    #### 2.1.1.2: Predictions of test_set
    test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(train_treated_model, train_control_model, test_treated_x, test_control_x)
    test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(train_treated_model, train_control_model, test_x)

    ### 2.1.2: Train models on all the data
    treated_model, control_model = t_model(treated_x, treated_y, control_x, control_y, model_class, model_params)
    #### 2.1.2.1: Predictions of all the data
    treated_y_pred, treated_y_prob, control_y_pred, control_y_prob = predictor_train_predictions(treated_model, control_model, treated_x, control_x)
    y_t1_pred, y_t0_pred, y_t1_prob, y_t0_prob, ite_prob, ite_pred = predictor_ite_predictions(treated_model, control_model, x)

    ## 2.2: S-Learner
    ### 2.1.1: Train models
    train_forest_model = RandomForestQuantileRegressor()
    train_forest_model.fit(train_xt, train_y.squeeze())

    ## 2.3: Transform the predictions
    if train_treated_y_prob is not None and not train_treated_y_prob.isna().all():
        test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
        train_set = pd.concat([train_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
        all_set = pd.concat([t, y_t1_pred, y_t1_prob, y_t0_pred, y_t0_prob, ite_pred, ite_prob, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1).copy()
        test_set.to_csv(f'output/csv2/{dataset}/SPECIAL_ test_set.csv', index=False)
        train_set.to_csv(f'output/csv2/{dataset}/SPECIAL_train_set.csv', index=False)

    else:
        test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
        train_set = pd.concat([train_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
        all_set = pd.concat([t, y_t1_pred, y_t0_pred, ite_pred, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1).copy()
        test_set.to_csv(f'output/csv2/{dataset}/NORMAL_ test_set.csv', index=False)
        train_set.to_csv(f'output/csv2/{dataset}/NORMAL_train_set.csv', index=False)


    ## 2.4: Transform the predictions of TWINS to continuous outcomes
    if dataset == "TWINSC":
        # Delete columns y_t1_pred and y_t0_pred, ite_pred
        # test_set = test_set.drop(columns=['y_t1_pred', 'y_t0_pred', 'ite_pred']).copy()

        test_set = test_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1).copy()
        # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
        test_set = test_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

        train_set = train_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1).copy()
        train_set = train_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

        all_set = all_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1).copy()
        # all_set = all_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

    test_set.to_csv(f'output/csv2/{dataset}/0_ test_set2.csv', index=False)
    train_set.to_csv(f'output/csv2/{dataset}/0_train_set2.csv', index=False)

    ## 2.5: Merge the train and test set
    all_data = pd.concat([train_set, test_set], ignore_index=True).copy()

    all_data.to_csv(f'output/csv2/{dataset}/0_ all_data.csv', index=False)

    ## 2.6: Add columns for costs performance measurement
    all_data['category'] = all_data.apply(categorize, axis=1, is_pred=False)
    all_data['Decision'] = all_data.apply(categorize_decision, axis=1)
    # all_data['category_pred'] = all_data.apply(categorize, axis=1)
    # all_data['category_rej'] = all_data.apply(categorize, axis=1)
    all_data['ite_mistake'] = all_data.apply(lambda row: False if row['ite_pred']==row['ite'] else True, axis=1)
    
    ## 2.7: Add columns for rejection performance measurement
    all_data['ite_reject'] = all_data.apply(lambda row: row['ite_pred'], axis=1)
    all_data['se'] = (all_data['ite'] - all_data['ite_pred']) ** 2
    all_data['decile'] = pd.qcut(all_data['ite'], q=10, labels=False, duplicates='drop')
    all_data['ite_sign'] = all_data['ite'].div(all_data['ite'].abs(), fill_value=0)
    all_data['ite_sign'] = all_data['ite_sign'].fillna(0)

    # export to csv
    all_data.to_csv(f'output/csv/{dataset}/0_{dataset}_all_data.csv', index=False)
    x.to_csv(f'output/csv/{dataset}/0_{dataset}_x.csv', index=False)
    xt.to_csv(f'output/csv/{dataset}/0_{dataset}_xt.csv', index=False)

    all_data_list.append(all_data)
    x_list.append(x)
    xt_list.append(xt)
    train_forest_model_list.append(train_forest_model)

# Chapter 3 & 4: Rejection Scores and Rejection

i = -1 # (dataset 0 first)
for dataset in datasets:
    # reset experiment_id for each dataset
    experiment_id = -2 
    metrics_results[dataset] = {}
    i += 1

    ## Output
    with open(file_path, 'a') as file:
        file.write(f"\nREJECTION for {dataset}\n\n")

    #######################################################################################################################
    # Rejection Architecture
    architecture="Separated Architecture"

    #######################################################################################################################
    # No Rejection
    experiment_id += 1
    experiment_name = "No Rejector - Baseline Model"
    experiment_names.update({experiment_id: f"{experiment_name}"})

    # Calculate the performance metrics
    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data_list[i], file_path, metrics_results, append_metrics_results=False, print=False)        
    metrics_results[dataset].update({experiment_id: metrics_dict})

    #######################################################################################################################
    # Type 0 - Perfect Rejection
    experiment_id += 1
    experiment_name =  "Perfect Rejection"
    abbreviation = "Perfect"
    experiment_names.update({experiment_id: f"{experiment_name}"})

    rr_perfect = []
    rmse_accepted_perfect = []
    rmse_rejected_perfect = []
    rmse_change_accepted_perfect = []
    
    all_data_list[i] = all_data_list[i].sort_values(by='se', ascending=False).copy()
    all_data_list[i] = all_data_list[i].reset_index(drop=True)

    for rr in range(1, max_rr*detail_factor):
        num_to_set = int(rr / (100.0*detail_factor) * len(all_data_list[i])) # example: 60/100 = 0.6 * length of the data

        all_data_list[i]['ite_reject'] = all_data_list[i]['ite_pred']
        all_data_list[i]['ite_reject'] = all_data_list[i]['ite_reject'].astype(object)  # Change dtype of entire column
        all_data_list[i].loc[:num_to_set -1, 'ite_reject'] = 'R'

        metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data_list[i], file_path)

        if metrics_result:
            rr_perfect.append(metrics_result.get('Rejection Rate', None))
            rmse_accepted_perfect.append(metrics_result.get('RMSE Accepted', None))
            rmse_rejected_perfect.append(metrics_result.get('RMSE Rejected', None))
            rmse_change_accepted_perfect.append((metrics_result.get('RMSE Accepted', None) - metrics_result.get('RMSE Original', None)) / metrics_result.get('RMSE Original', None) * 100)
        else:
            rr_perfect.append(None)
            rmse_accepted_perfect.append(None)
            rmse_rejected_perfect.append(None)

    # export to csv
    data = {
        'Reject Rate': rr_perfect,
        'RMSE Accepted': rmse_accepted_perfect,
        'RMSE Rejected': rmse_rejected_perfect,
        'Improvement': rmse_change_accepted_perfect
    }

    df = pd.DataFrame(data)
    df.to_csv(f'output/csv/{dataset}/0_{dataset}_perfect.csv', index=False)

    # Graph with reject rate and rmse_accepted & rmse_rejected
    twolinegraph(rr_perfect, "Reject Rate", rmse_accepted_perfect, "RMSE of Accepted Samples", "green", rmse_rejected_perfect, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse.pdf")
    onelinegraph(rr_perfect, "Reject Rate", rmse_accepted_perfect, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.pdf")
    onelinegraph(rr_perfect, "Reject Rate", rmse_rejected_perfect, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.pdf")

    # optimal model
    min_rmse = min(rmse_accepted_perfect)  # Find the minimum
    min_rmse_index = rmse_accepted_perfect.index(min_rmse)  # Find the index of the minimum RMSE
    optimal_reject_rate = rr_perfect[min_rmse_index]  # Get the rejection rate at the same index

    all_data_list[i]['ite_reject'] = all_data_list[i]['ite_pred']
    all_data_list[i]['ite_reject'] = all_data_list[i]['ite_reject'].astype(object)  # Change dtype of entire column
    all_data_list[i].loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data_list[i], file_path, metrics_results, append_metrics_results=False, print=False)
    metrics_results[experiment_id] = metrics_dict
    list_results = []
    
    reject_rates_list[dataset] = {}
    rmse_rank_accepted_list[dataset] = {}
    rmse_rank_weighted_accepted_list[dataset] = {}
    sign_error_accepted_list[dataset] = {}
    signerror_weighted_accepted_list[dataset] = {}
    experiment_ids_list[dataset] = {}
    rmse_accepted_list[dataset] = {}
    rmse_change_accepted_list[dataset] = {}
    heuristic_cutoff_list[dataset] = {}
    sign_error_change_accepted_list[dataset] = {}
    rmse_rank_change_accepted_list[dataset] = {}
    rmse_rank_weighted_change_accepted_list[dataset] = {}
    metrics_results_list_global[dataset] = {}

    # Type 1
    for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
        experiment_id += 1
        experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type I"
        metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = novelty_rejection(1, max_rr, detail_factor, model, x_list[i], all_data_list[i], file_path, experiment_id, dataset, rmse_accepted_perfect)
        metrics_results[dataset].update({experiment_id: metrics_dict})
        reject_rates_list[dataset].update({experiment_id: reject_rates})
        heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
        metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
        experiment_ids_list[dataset].update({experiment_id: experiment_id})
    
    # Type 2
    for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
        experiment_id += 1
        experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type II"
        metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = novelty_rejection(2, max_rr, detail_factor, model, x_list[i], all_data_list[i], file_path, experiment_id, dataset, rmse_accepted_perfect)
        metrics_results[dataset].update({experiment_id: metrics_dict})
        reject_rates_list[dataset].update({experiment_id: reject_rates})
        heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
        metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
        experiment_ids_list[dataset].update({experiment_id: experiment_id})

    # Type 3
    for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
        experiment_id += 1
        experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type III"
        metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = novelty_rejection(3, max_rr, detail_factor, model, x_list[i], all_data_list[i], file_path, experiment_id, dataset, rmse_accepted_perfect)
        metrics_results[dataset].update({experiment_id: metrics_dict})
        reject_rates_list[dataset].update({experiment_id: reject_rates})
        heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
        metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
        experiment_ids_list[dataset].update({experiment_id: experiment_id})

    #######################################################################################################################
    # Ambiguity
    #######################################################################################################################

    # Type 1
    experiment_id += 1
    model = "RandomForestQuantileRegressor"
    abbreviation = "RFQR"
    experiment_names[experiment_id] = f"Rejection based on RandomForestQuantileRegressor - Ambiguity Type I"
    
    # def ambiguity_rejection(max_rr, detail_factor, model, xt, all_data, file_path, experiment_id, dataset, rmse_accepted_perfect=[]):
    max_rr, detail_factor, model, xt, all_data, file_path, experiment_id, dataset, rmse_accepted_perfect = max_rr, detail_factor, train_forest_model_list[i], xt_list[i], all_data_list[i], file_path, experiment_id, dataset, rmse_accepted_perfect
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
    print("Size of CI:")
    print(size_of_ci)
    all_data['Ambiguity Score'] = size_of_ci
    all_data = all_data.sort_values(by='Ambiguity Score', ascending=False).copy()
    all_data = all_data.reset_index(drop=True)

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

    # # Return values
    # return metrics_dict, reject_rates, heuristic_reject_rate, metrics_results_list
    
    # metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = ambiguity_rejection(max_rr, detail_factor, train_forest_model_list[i], xt_list[i], all_data_list[i], file_path, experiment_id, dataset, rmse_accepted_perfect)

    metrics_results[dataset].update({experiment_id: metrics_dict})
    reject_rates_list[dataset].update({experiment_id: reject_rates})
    heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
    metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
    experiment_ids_list[dataset].update({experiment_id: experiment_id})


# CHAPTER 5: Report Performance Evaluation
canvas_change_loop(reject_rates_list, metrics_results_list_global, "RMSE Change (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'rmse', -11, 3, f'Impact of Rejection on the RMSE of the TE', datasets)
canvas_change_loop(reject_rates_list, metrics_results_list_global, "Similarity 50% Improved (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Similarity 50% Deviation from No-Rejection (%)', 'similarity', -4, 6, f'Impact of Rejection on the similarity (50%) Metric', datasets)
canvas_change_loop(reject_rates_list, metrics_results_list_global, "Achieved Result by top 50% Improved (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Achieved result 50% Deviation from No-Rejection (%)', 'achievedresult', -40, 40, f'Impact of Rejection on the Achieved Result (50%) Metric', datasets)

# Old:
# canvas_change_loop(reject_rates_list, metrics_results_list_global, "Sign Accuracy change (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Sign Accuracy Deviation from No-Rejection (%)', 'signaccuracy', -30, 80, f'Impact of Rejection on the Sign Accuracy of the TE', datasets)
# canvas_change_loop(reject_rates_list, metrics_results_list_global, "Weighted Sign Accuracy change (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Weighted Sign Accuracy Deviation from No-Rejection (%)', 'weightedsignaccuracy',  -30, 80, f'Impact of Rejection on Weighted Sign Accuracy of the TE', datasets)
# canvas_change_loop(reject_rates_list, metrics_results_list_global, "Positive Potential Accuracy change (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Positive Potential Accuracy Deviation from No-Rejection (%)', 'positivepotential', -10, 10, f'Impact of Rejection on the Positive Potential Accuracy', datasets)
# canvas_change_loop(reject_rates_list, metrics_results_list_global, "Adverse Effect Accuracy change (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Positive Potential Accuracy Deviation from No-Rejection (%)', 'adverseeffect', -10, 10, f'Impact of Rejection on the Positive Potential Accuracy', datasets)

i = -1
for dataset in datasets:
    i += 1
    canvas_change(reject_rates_list[dataset],"Achieved Result by top 10% Improved (%)", "Achieved Result by top 20% Improved (%)", "Achieved Result by top 30% Improved (%)", "Achieved Result by top 40% Improved (%)", "Achieved Result by top 50% Improved (%)", metrics_results_list_global[dataset], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','Achieved Result (%)', 'achievedresult', -100, 50, f'Impact of Rejection ({dataset})')

    # OLD CODE:
    # plot_summary(reject_rates_list[dataset], rmse_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Accepted", "RMSEAccepted")
    # plot_summary(reject_rates_list[dataset], [result.get('RMSE Accepted', None) for result in metrics_results_list_global[dataset][2]], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Accepted", "RMSEAccepted")
    # plot_summary(reject_rates_list[dataset], rmse_rank_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Rank Accepted", "RMSERankAccepted")
    # plot_summary(reject_rates_list[dataset], sign_error_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on Sign Error Accepted", "SignErrorAccepted")
    # plot_summary(reject_rates_list[dataset], rmse_rank_weighted_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Rank Weighted Accepted", "RMSERankWeightedAccepted")
    # plot_canvas(reject_rates_list[dataset], rmse_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Accepted", "RMSEAccepted")
    # # 9x9 plots:
    # canvas_change(reject_rates_list[dataset], [result.get('RMSE Change (%)', None) for result in metrics_results_list_global[dataset]], [result.get('RMSE Change (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'rmse', -9, 3, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    
    # canvas_change(reject_rates_list[dataset],"Similarity 50% Accepted (%)", "Similarity 50% Rejected (%)", metrics_results_list_global[dataset], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)',"Similarity (%)", 'similarity', 0, 100, f'Impact of Rejection ({dataset})')
    # canvas_change(reject_rates_list[dataset],"Sign Accuracy Accepted (%)", "Sign Accuracy Rejected (%)", metrics_results_list_global[dataset], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','Sign Accuracy (%)', 'signaccuracy', 0, 100, f'Impact of Rejection ({dataset})')
    # canvas_change(reject_rates_list[dataset],"Weighted Sign Accuracy Accepted (%)", "Weighted Sign Accuracy Rejected (%)", metrics_results_list_global[dataset], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','Weighted Sign Accuracy (%)', 'weightedsignaccuracy', 0, 100, f'Impact of Rejection ({dataset})')
    # canvas_change(reject_rates_list[dataset],"Adverse Effect Accuracy Accepted (%)", "Adverse Effect Accuracy Rejected (%)", metrics_results_list_global[dataset], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','Adverse Effect Accuracy (%)', 'adverseeffect', 0, 100, f'Impact of Rejection ({dataset})')
    # canvas_change(reject_rates_list[dataset],"Positive Potential Accuracy Accepted (%)", "Positive Potential Accuracy Rejected (%)", metrics_results_list_global[dataset], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','Positive Potential Accuracy (%)', 'positivepotential', 0, 100, f'Impact of Rejection ({dataset})')

    # canvas_change(reject_rates_list[dataset], [result.get('Similarity 50% Accepted (%)', None) for result in metrics_results_list_global[dataset]], [result.get('Similarity 50% Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'signaccuracy', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    # canvas_change(reject_rates_list[dataset], [result.get('Sign Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]], [result.get('Sign Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'similarity', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    # canvas_change(reject_rates_list[dataset], [result.get('Weighted Sign Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]],[result.get('Weighted Sign Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'weightedsignaccuracy', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    # canvas_change(reject_rates_list[dataset], [result.get('Adverse Effect Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]],[result.get('Adverse Effect Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'adverseeffect', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    # canvas_change(reject_rates_list[dataset], [result.get('Positive Potential Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]],[result.get('Positive Potential Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'positivepotential', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')

    # Chapter 8: Output to file
    with open(file_path, 'a') as file:

        file.write("\n\nTable of all_data (First 5 rows)\n")
        file.write(tabulate(all_data_list[i].head(5), headers='keys', tablefmt='pretty', showindex=False))
        
        file.write ("\n")
        for exp_number, description in experiment_names.items():
            file.write(f"# Experiment {exp_number}: {description}\n")

        file.write("\nTable of results of the experiments\n")
        df = pd.DataFrame(metrics_results[dataset])
        df = df.transpose() 
        file.write(tabulate(df.transpose(), headers='keys', tablefmt='rounded_grid', showindex=True))

import matplotlib.pyplot as plt
twinsc_rmse = {} 
twinsc_reject_rates = {} 

# Define the data for TWINSC dataset
for experiment_id, reject_rates in reject_rates_list['TWINSC'].items():
    twinsc_reject_rates[experiment_id] = reject_rates

for i in range(1, len(reject_rates_list['TWINSC']) + 1):  # Adjusting range to start from 1
    twinsc_rmse[i] = [result.get('RMSE Accepted', None) for result in metrics_results_list_global.get('TWINSC', {}).get(i, {})]

ihdp_reject_rates = {}
ihdp_rmse = {}

# Define the data for TWINSC dataset
for experiment_id, reject_rates in reject_rates_list['IHDP'].items():
    ihdp_reject_rates[experiment_id] = reject_rates

for i in range(1, len(reject_rates_list['IHDP']) + 1):  # Adjusting range to start from 1
    ihdp_rmse[i] = [result.get('RMSE Accepted', None) for result in metrics_results_list_global.get('IHDP', {}).get(i, {})]

# Create the canvas with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the data for TWINSC dataset
for i in range(1, len(twinsc_rmse) + 1):
    ax1.plot(twinsc_reject_rates[i], twinsc_rmse[i], label=f'Experiment {i}')
ax1.set_xlabel('Reject Rate')
ax1.set_ylabel('RMSE')
ax1.set_title('Impact of Rejection on the RMSE of the TE (TWINS)')
ax1.legend()  # Show legend

# Plot the data for IHDP dataset
for i in range(1, len(ihdp_rmse) + 1):
    ax2.plot(ihdp_reject_rates[i], ihdp_rmse[i], label=f'Experiment {i}')
ax2.set_xlabel('Reject Rate')
ax2.set_ylabel('RMSE')
ax2.set_title('Impact of Rejection on the RMSE of the TE (IHDP)')

# Adjust the layout and display the plot
plt.tight_layout()

# Save the plot to a file in the specified folder
plt.savefig(f"{folder_path}overleaf/rmserejectcurve.pdf")
plt.close()

# metrics_results[dataset] = pd.DataFrame.from_dict(metrics_results[dataset], orient='index')

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")