from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import pandas as pd

# Function to train a model equal to the one in one_class_classification_rejector.py
def train_model(x, model_class, **model_params):
    # Train the model
    model = model_class(**model_params)
    model.fit(x)
    return model

def f(type, contamination, t_x, ut_x, t_data, ut_data, detail_factor, model_name, all_data):
    contamination /= (100 * detail_factor)  # max of 0.5
    if model_name == IsolationForest:
        t_model = train_model(t_x, IsolationForest, contamination=contamination, random_state=42)
        ut_model = train_model(ut_x, IsolationForest, contamination=contamination, random_state=42)
    elif model_name == OneClassSVM:
        t_model = train_model(t_x, OneClassSVM, nu=contamination)
        ut_model = train_model(ut_x, OneClassSVM, nu=contamination)
    elif model_name == LocalOutlierFactor:
        t_model = train_model(t_x, LocalOutlierFactor, contamination=contamination, novelty=True)
        ut_model = train_model(ut_x, LocalOutlierFactor, contamination=contamination, novelty=True)

    if type == 2:
        t_data['ood'] = pd.Series(ut_model.predict(t_x), name='ood').copy()
        ut_data['ood'] = pd.Series(t_model.predict(ut_x), name='ood').copy()
        all_data['amount_of_times_rejected_new'] = all_data.apply(lambda row: 1 if row['ood'] == -1 else 0, axis=1)
    if type == 3:
        ut_data['ood-ut'] = pd.Series(t_model.predict(ut_x), name='ood').copy()
        ut_data['ood-t'] = pd.Series(t_model.predict(ut_x), name='ood').copy()
        # ut_data['ood'] = (ut_data['ood-ut'] + ut_data['ood-t']) / 2
        ut_data['ood'] = ut_data[['ood-ut', 'ood-t']].max(axis=1)

    all_data = pd.concat([t_data, ut_data], ignore_index=True).copy()
    all_data['amount_of_times_rejected_new'] = all_data.apply(lambda row: 1 if row['ood'] == -1 else 0, axis=1)
    return all_data['amount_of_times_rejected_new']