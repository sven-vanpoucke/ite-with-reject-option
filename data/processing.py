
# Import of the packages.
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t):
  """
  this function splits the data into treated and control groups
  """
  # for training data
  train_treated_x = train_x[train_t['treatment'] == 1]
  train_control_x = train_x[train_t['treatment'] == 0]
  train_treated_y = train_y[train_t['treatment'] == 1]
  train_control_y = train_y[train_t['treatment'] == 0]
  
  # for test data
  test_treated_x = test_x[test_t['treatment'] == 1]
  test_control_x = test_x[test_t['treatment'] == 0]
  test_treated_y = test_y[test_t['treatment'] == 1]
  test_control_y = test_y[test_t['treatment'] == 0]

  return train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y

def merge_test_train(train_treated_x, train_treated_y, train_control_x, train_control_y, test_treated_x, test_treated_y, test_control_x, test_control_y, train_x, train_t, train_y, test_x, test_t, test_y, train_ite, test_ite, train_potential_y, test_potential_y, x_scaling):
    """
    This function merges the test and train data
    """
    
    ## Merge test_set & the train_set
    treated_x = pd.concat([train_treated_x, test_treated_x], ignore_index=True).copy() # Under each other
    treated_y = pd.concat([train_treated_y, test_treated_y], ignore_index=True).copy() # Under each other
    control_x = pd.concat([train_control_x, test_control_x], ignore_index=True).copy() # Under each other
    control_y = pd.concat([train_control_y, test_control_y], ignore_index=True).copy() # Under each other
    x = pd.concat([train_x, test_x], ignore_index=True).copy()  # Under each other

    t = pd.concat([train_t, test_t], ignore_index=True).copy() # Under each other
    xt = pd.concat([x, t], axis=1).copy() # Left & right from eachother

    train_xt = pd.concat([train_x, train_t], axis=1).copy() # Left & right from eachother
    test_xt = pd.concat([test_x, test_t], axis=1).copy()# Left & right from eachother

    y = pd.concat([train_y, test_y], ignore_index=True).copy() # Under each other
    y = pd.DataFrame(y)
    ite = pd.concat([train_ite, test_ite], ignore_index=True).copy() # Under each other
    potential_y = pd.concat([train_potential_y, test_potential_y], ignore_index=True).copy() # Under each other

    if x_scaling:
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    return treated_x, treated_y, control_x, control_y, x, t, xt, train_xt, test_xt, y, ite, potential_y