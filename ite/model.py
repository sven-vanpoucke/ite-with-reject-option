from sklearn.linear_model import LogisticRegression
import pandas as pd
from tabulate import tabulate

# T-model
def t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params):
    # Training separate models for treated and control groups
    treated_model = model_class(**model_params)
    treated_model.fit(train_treated_x, train_treated_y.values.flatten())

    control_model = model_class(**model_params)
    control_model.fit(train_control_x, train_control_y.values.flatten())
    
    return treated_model, control_model
