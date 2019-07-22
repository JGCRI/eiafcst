"""
Produce plots for evaluating a trained model.

Usage:
plot_model_predictions.py model outdir zerovars [prefix]
    model:      Full path to saved model
    outdir:     Full path to plot output directory
    zerovars:   Comma-separated variables to zero out (one of gas, petrol, elec, gdp_prev)
    prefix:     Output plot file prefix (optional)

Caleb Braun
7/2/19
"""
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import sys
import os

from eiafcst.models.model_gdp import load_inputs


def run_prediction(model, full_data):
    """
    Get predictions from the model with known results.

    Returns GDP predictions and autoencoder predictions.
    """
    dsize = len(full_data['elec'])
    preds = np.empty(dsize)
    pdecs = []
    for i in range(dsize):
        ele = full_data['elec'][i]
        gas = full_data['gas'][i]
        pet = full_data['petrol'][i]

        nwk = ele.shape[0]
        ts = np.arange(full_data['time'][i], full_data['time'][i] + 1, 1 / nwk)
        gdp_prev = np.repeat(full_data['gdp_prev'][i], nwk)

        p_outputs = model.predict([ele, ts, gdp_prev, gas, pet])
        pgdp = p_outputs[0].mean()  # GDP prediction
        pdec = p_outputs[1]  # Decoder prediction

        pgdp = pgdp * full_data['gdp'].std() + full_data['gdp'].mean()
        preds[i] = pgdp

        pdecs.append(pdec)

    return preds, pdecs


def combine_model_inputs(varname, full_data_order, train, dev=None, test=None):
    """Combine the inputs, putting them in correct time order."""
    full_var = []
    for v in train[varname]:
        full_var.append(v)
    if dev is not None:
        for v in dev[varname]:
            full_var.append(v)
    if test is not None:
        for v in test[varname]:
            full_var.append(v)

    full_var = np.array(full_var)
    return full_var[full_data_order]


def plot_pred_v_true(preds, inputs, c1, c2, dot_size, out, prefix, title_prefix):
    """
    Plot full time series of predicted values vs true values.

    :param preds:           Predictions
    :param inputs:          Model inputs
    :param c1, c2:          Plot colors
    :param dot_size:        Plot dot size int points
    :param out:             Output directory (needs trailing slash)
    :param prefix:          Model name
    :param title_prefix:    Prefix for plot title
    """
    outf = os.path.join(out, f'{prefix}_predictions_{title_prefix.split()[0].lower()}.png')

    preds_data_order = np.argsort(inputs['time'])
    x = inputs['time'][preds_data_order]
    y1 = preds[preds_data_order]
    y2 = inputs['gdp'][preds_data_order]

    plt.scatter(x, y1, color=c1, s=dot_size)
    plt.scatter(x, y2, color=c2, s=dot_size)
    plt.plot(x, y1, color=c1, label='Predictions')
    plt.plot(x, y2, color=c2, label='True Values')
    plt.xlabel('Quarters since 2006Q1')
    plt.ylabel('Billion $USD')
    plt.title(f'{title_prefix}Predictions vs. Actual for {prefix}')
    plt.legend()
    plt.savefig(outf)
    plt.clf()


def usage():
    """Show file usage."""
    print("Usage:\nplot_model_predictions.py model outdir zerovars [prefix]")
    print("\tmodel:\t\tFull path to saved model")
    print("\toutdir:\t\tFull path to plot output directory")
    print("\tzerovars:\tComma-separated variables to zero out (none or one of gas, petrol, elec, gdp_prev)")
    print("\tprefix:\t\tOutput plot file prefix")
    raise SystemExit


def main():
    """Load and run predictions on the model, plot outputs."""
    try:
        model_name = sys.argv[1]
        out = sys.argv[2]
        removevars = sys.argv[3].split(',')
    except IndexError:
        usage()

    try:
        prefix = sys.argv[4]
    except IndexError:
        prefix = os.path.basename(model_name).split('.')[0]

    model = keras.models.load_model(model_name)

    train, dev = load_inputs(['train', 'dev'])

    # Optionally, zero out certain variables for analysis
    if removevars != ['none']:
        for rv in removevars:
            train[rv] = np.array([np.zeros(a.shape) for a in train[rv]])
            dev[rv] = np.array([np.zeros(a.shape) for a in dev[rv]])

    full_data_order = np.argsort(np.concatenate([train['time'], dev['time']]))
    full_data = {
        'gas': combine_model_inputs('gas', full_data_order, train, dev),
        'petrol': combine_model_inputs('petrol', full_data_order, train, dev),
        'elec': combine_model_inputs('elec', full_data_order, train, dev),
        'time': combine_model_inputs('time', full_data_order, train, dev),
        'gdp': combine_model_inputs('gdp', full_data_order, train, dev),
        'gdp_prev': combine_model_inputs('gdp_prev', full_data_order, train, dev)
    }

    # Plot constants
    c1 = '#083D77'
    c2 = '#F95738'
    dot_size = 14  # size in points

    print(f'Saving plots to {out}')

    # Plot full time series predicted v true for training data
    preds, _ = run_prediction(model, train)
    plot_pred_v_true(preds, train, c1, c2, dot_size, out, prefix, 'Training Set ')

    # Plot full time series predicted v true for training data
    preds, _ = run_prediction(model, dev)
    plot_pred_v_true(preds, dev, c1, c2, dot_size, out, prefix, 'Development Set ')

    # Plot full time series predicted v true
    preds, decoder = run_prediction(model, full_data)
    plot_pred_v_true(preds, full_data, c1, c2, 0, out, prefix, 'Full ')

    # Plot the decoder predictions
    outf = os.path.join(out, f'{prefix}_decoder_predictions.png')
    q = 18  # Pick a quarter to plot
    z1 = full_data['elec'][q]
    z2 = decoder[q]
    plt.plot(np.mean(z1, axis=(2))[0, :], label='True Values')
    plt.plot(np.mean(z2, axis=(2))[0, :], label='Predictions')
    plt.xlabel(f'Quarters since {2006 + q / 4}')
    plt.ylabel('Standardized electric load')
    plt.legend()
    plt.savefig(outf)
    plt.clf()


if __name__ == '__main__':
    main()
