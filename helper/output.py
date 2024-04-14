import pandas as pd
from datetime import datetime

def helper_output(dataset, folder_path='output/'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_print = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f'results_{dataset}_{timestamp}.txt'
    file_path = folder_path + filename

    with open(file_path, 'a') as file:
        file.write(f"CHAPTER 1: INTRODUCTION\n")
        file.write(f"# This section introduces the purpose and background of the analysis.\n\n")
        file.write("In this analysis, we aim to evaluate the performance of different reject options for Information Treatment Effect (ITE) models.") 
        file.write("The ITE model predicts the individual treatment effects in a given dataset, providing valuable insights into the impact of interventions.\n")
        file.write(f"For your information, this file has been automatically generated on: {timestamp_print}\n")
    return timestamp, filename, file_path

def helper_output_loop(folder_path='output/'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_print = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f'results_{timestamp}.txt'
    file_path = folder_path + filename

    with open(file_path, 'a') as file:
        file.write(f"CHAPTER 1: INTRODUCTION\n")
        file.write(f"# This section introduces the purpose and background of the analysis.\n\n")
        file.write("In this analysis, we aim to evaluate the performance of different reject options for Information Treatment Effect (ITE) models.") 
        file.write("The ITE model predicts the individual treatment effects in a given dataset, providing valuable insights into the impact of interventions.\n")
        file.write(f"For your information, this file has been automatically generated on: {timestamp_print}\n")
    return timestamp, filename, file_path