"""Read in a model save file and write a table of actual and predicted GDP for all years.

Usage:

model_predict.py modelfile

"""


from tensorflow import keras
import numpy as np
import sys
import os

from eiafcst.models.model_gdp import load_inputs, run_prediction
import eiafcst.models.diagnostic.plot_model_predictions as predict

def get_prediction_and_data(filename):

    model = keras.models.load_model(filename)

    train, dev, test = load_inputs(['train','dev','test'])

    data_order = np.argsort(np.concatenate([train['time'], dev['time'], test['time']]))

    alldata = {
        'gas': predict.combine_model_inputs('gas', data_order, train, dev, test),
        'petrol': predict.combine_model_inputs('petrol', data_order, train, dev, test),
        'elec': predict.combine_model_inputs('elec', data_order, train, dev, test),
        'time': predict.combine_model_inputs('time', data_order, train, dev, test),
        'gdp': predict.combine_model_inputs('gdp', data_order, train, dev, test),
        'gdp_prev': predict.combine_model_inputs('gdp_prev', data_order, train, dev, test)
    }

    preds, _ = predict.run_prediction(model, alldata)

    data_order = np.argsort(alldata['time'])
    qtr = alldata['time'][data_order]
    predicted = preds[data_order]
    actual = alldata['gdp'][data_order]

    return(qtr, predicted, actual)
    

if __name__ == '__main__':

    filename = sys.argv[1]
    (qtr, predicted, actual) = get_prediction_and_data(filename)

    sys.stdout.write('qtr, gdp_predicted, gdp_actual\n')
    for i in range(len(qtr)):
        sys.stdout.write(f'{qtr[i]}, {predicted[i]}, {actual[i]}\n')

    

