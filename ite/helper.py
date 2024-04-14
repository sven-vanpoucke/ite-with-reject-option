from evaluator.cost import calculate_misclassification_cost

def improvement(old_value, new_value):
    old_value = float(old_value)
    new_value = float(new_value)
    improvement = ((new_value-old_value)/new_value*100) if new_value != 0 else 0
    return improvement


def print_reject_improvements(file_path, rr_2, total_cost_ite, total_cost_ite_2, accurancy, accurancy_2, micro_distance_threedroc, micro_distance_threedroc_2, macro_distance_threedroc, macro_distance_threedroc_2):
    cost_improvement = improvement(total_cost_ite, total_cost_ite_2)
    accurancy_improvement = improvement(accurancy, accurancy_2)
    micro_distance_improvement = improvement(micro_distance_threedroc, micro_distance_threedroc_2)
    macro_distance_improvement = improvement(macro_distance_threedroc, macro_distance_threedroc_2)

    with open(file_path, 'a') as file:
        # Write the total misclassification cost after probability rejection
        file.write(f"\nImprovements due to a rejection rate of {(rr_2*100):.2f}%:\n")

        # Write the total misclassification cost
        file.write(f'- Change of the Misclassification Cost: {cost_improvement:.2f}%\n')
        file.write(f'- Change of the ITE Accurancy: {accurancy_improvement:.2f}%\n')
        file.write(f'- Change of the 3D ROC (micro): {micro_distance_improvement:.2f}%\n')
        file.write(f'- Change of the 3D ROC (macro): {macro_distance_improvement:.2f}%\n')


def print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc):
    # Calculate total misclassification cost
    total_cost_ite_2 = calculate_misclassification_cost(test_set, 2)


    # accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)

    # print_reject_improvements(file_path, rr_2, total_cost_ite, total_cost_ite_2, accurancy, accurancy_2, micro_distance_threedroc, micro_distance_threedroc_2, macro_distance_threedroc, macro_distance_threedroc_2)