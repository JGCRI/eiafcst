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
    trv = np.repeat("TRAIN", len(train["time"]))
    dv  = np.repeat("DEV", len(dev["time"]))
    tev = np.repeat("TEST", len(test["time"]))
    data_source = np.concatenate((np.repeat('TRAIN', len(train['time'])),
                                  np.repeat('DEV', len(dev['time'])),
                                  np.repeat('TEST', len(test['time']))))
    data_source = data_source[data_order]


    alldata = {
        'gas': predict.combine_model_inputs('gas', data_order, train, dev, test),
        'petrol': predict.combine_model_inputs('petrol', data_order, train, dev, test),
        'elec': predict.combine_model_inputs('elec', data_order, train, dev, test),
        'time': predict.combine_model_inputs('time', data_order, train, dev, test),
        'gdp': predict.combine_model_inputs('gdp', data_order, train, dev, test),
        'gdp_prev': predict.combine_model_inputs('gdp_prev', data_order, train, dev, test),
    }

    gdpmean = alldata['gdp'].mean()
    gdpstd = alldata['gdp'].std()
    all_labels = (alldata['gdp'] - gdpmean) / gdpstd

    preds, resids, _ = run_prediction(model, alldata, 'All Data', all_labels, normal_mode=True)

    do_old = data_order
    data_order = np.argsort(alldata['time'])

    qtr = alldata['time'][data_order]
    predicted = preds[data_order]
    actual = alldata['gdp'][data_order]

    return(qtr, predicted, actual, data_source)
    

if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.stderr.write(f'Usage:  {sys.argv[0]} model_file output_file.\n')
        sys.exit(1)
    filename = sys.argv[1]
    (qtr, predicted, actual, source) = get_prediction_and_data(filename)

    outfile = open(sys.argv[2], 'w')
    outfile.write('qtr, gdp_predicted, gdp_actual, data_source\n')
    for i in range(len(qtr)):
        outfile.write(f'{qtr[i]}, {predicted[i]}, {actual[i]}, {source[i]}\n')

    

