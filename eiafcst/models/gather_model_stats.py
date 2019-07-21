"""Read in one or more model save files and write their stats into an output file.

Usage:
gather_model_stats.py modelfiles

"""

from tensorflow import keras
import numpy as np
import sys
import os

from eiafcst.models.model_gdp import load_inputs
import eiafcst.models.diagnostic.plot_model_predictions as predict

## Maximum number of convolutional layer parameter sets to output
MAXCONV = 5

def load_model_data(modelfile):
    """Load a saved model and pull some basic information

    :param modelfile: Name of the file holding the saved model
    :return: Tuple of useful model data:
                (keras model structure,
                 list of model layers,
                 list of layer names)
    """
    model = keras.models.load_model(modelfile) 
    layers = model.layers
    layer_names = [layer.get_config()['name'] for layer in layers]

    return (model, layers, layer_names)


def gather_model_config(modelstruct):
    """Gather model configuration parameters from a saved model.

    :param modelstruct: Model structure from load_model_data.
    :return: Tuple: (number of convolutional layers in downsampling branch,
                     list of convolutional parameters 
                          (width, number of filters, pool size),
                     list of dense layer widths (L1, L2, Lgdp) 
    """

    (model, layers, layer_names) = modelstruct

    ## Find all convolution layers.  Unfortunately, this includes some
    ## that are used in the upsampling, but we will deal with that in
    ## a minute.
    convlayers = [layer for layer in layers if isinstance(layer, keras.layers.Convolution1D) ]
    convlayer_cfgs = []
    for convlayer in convlayers:
        childs = next_layers(convlayer)
        ## If there is more than one child, then this is not a layer we are looking for.
        ## Also, if the child is not a max pooling layer, it's not one we're looking for. 
        if len(childs) == 1 and isinstance(childs[0], keras.layers.MaxPooling1D):
            maxpool = childs[0]
            ## Extract the configuration, considering the convolution and the  pooling
            ## as a single layer
            convlayer_cfgs.append(conv_config_extract(convlayer, maxpool))

    ## Now all the downsamplings are captured in convlayer_cfgs
    nconv = len(convlayer_cfgs)

    ## Extract the width of the following dense layers
    dense_names = ['dense', 'FinalEncoding', 'GDP_Hidden', 'GDP_Hidden2']
    dense_widths = [get_dl_width(name, layer_names, layers) for name in dense_names]

    return (nconv, convlayer_cfgs, dense_widths)
            

def conv_config_extract(convlayer, maxpool):
    """Return a tuple of parameters for a convolutional layer

    The convolutional layer as we define it consists of a keras convolutional layer and
    a subbsequent max pooling layer.  The parameters are:
    (kernel-width, number of filters, pool size)
    """

    cconf = convlayer.get_config()
    ks = cconf['kernel_size'][0]
    nf = cconf['filters']
    ps = maxpool.get_config()['pool_size'][0]

    return (ks, nf, ps)

def next_layers(layer):
    """Given a layer, find all layers that take input from this layer"""
    nodes = layer.outbound_nodes

    return [node.outbound_layer for node in nodes]

def get_dl_width(name, names, layers):
    """Find the width of a dense layer with the given name.
    
    :param name: Name of the target layer
    :param names: List of all layer names
    :param layers: List of all layers.

    This subroutine assumes the names are unique.
    """
    idx = [i for (i, val) in enumerate(names)]
    if len(idx) == 0:
        return 'NA'
    else:
        idx = idx[0]

    return layers[idx].output_shape[1]


def get_performance(model, dataset):
    """Get the model performance (root mean squared error) on a specified dataset

    :param model: Keras model structure
    :param dataset: Name of the dataset to evaluate against ("train", "dev", or "test")
    """
    
    [dat] = load_inputs([dataset])

    ## Not strictly needed, but sticking close to original usage in case there are hidden
    ## assumptions.
    data_order = np.argsort(dat['time'])

    alldata = {
        'gas': predict.combine_model_inputs('gas', data_order, dat),
        'petrol': predict.combine_model_inputs('petrol', data_order, dat),
        'elec': predict.combine_model_inputs('elec', data_order, dat),
        'time': predict.combine_model_inputs('time', data_order, dat),
        'gdp': predict.combine_model_inputs('gdp', data_order, dat),
        'gdp_prev': predict.combine_model_inputs('gdp_prev', data_order, dat)
    }

    pgdp, pdecoder = predict.run_prediction(model, alldata)
    trugdp = alldata['gdp'][data_order]
    resids = (pgdp[data_order] - trugdp)/trugdp
    meansqrresid = np.mean(resids*resids)
    
    return np.sqrt(meansqrresid)

def convparm2str(convparm):
    """Convert convolutional layer plans to a comma separated string"""
    return ', '.join([str(x) for x in convparm])


def model_stats(filename):

    modelstruct = load_model_data(filename)
    model = modelstruct[0]
    perf_train = get_performance(model, 'train')
    perf_dev = get_performance(model, 'dev')
    perf_test = get_performance(model, 'test')

    nconv, convparm, dlparm = gather_model_config(modelstruct)

    ## Format this into a string that will form one row in the output table.
    ## Columns are:
    ## runid, rmstrain, rmsdev, rmstest, nconv, width1, nfilt1, pool1, ... width5, nfilt5, pool5

    runid, _ = os.path.splitext(os.path.basename(filename))

    ## If there are fewer than 5 convolutional layers, fill in the rest with NA values
    for i in range(nconv, MAXCONV):
        convparm.append(['NA', 'NA', 'NA'])

    allconvparms = ', '.join([convparm2str(parms) for parms in convparm])

    return f'{runid}, {perf_train}, {perf_dev}, {perf_test}, {nconv}, {allconvparms}\n'


if __name__ == '__main__':
    
    filenames = sys.argv[1:]

    ## Write to stdout; redirect to go to file.
    sys.stdout.write('runid, perf_train, perf_dev, perf_test, nconv')
    for i in range(1, MAXCONV+1):
        sys.stdout.write(f', width{i}, nfilt{i}, npool{i}')
    sys.stdout.write('\n')

    for filename in filenames:
        sys.stdout.write(model_stats(filename))



