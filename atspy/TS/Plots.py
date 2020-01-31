# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from io import BytesIO
import base64


SIGNAL_COLOR='green'
FORECAST_COLOR='blue'
RESIDUE_COLOR='red'
COMPONENT_COLOR='navy'
SHADED_COLOR='turquoise'
UPPER_COLOR='grey'
LOWER_COLOR='black'


def add_patched_legend(ax , names):
    # matplotlib does not like labels starting with '_'
    patched_names = []
    for name in names:
        # remove leading '_' => here, this is almost OK: no signal transformation
        patched_name = name[2:] if(name.startswith('__')) else name
        patched_name = patched_name[1:] if(patched_name.startswith('_')) else patched_name
        patched_names = patched_names + [ patched_name ]
    ax.legend(patched_names)

def decomp_plot(df, time, signal, estimator, residue, name = None, format='png', max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)


    import matplotlib
    # print("MATPLOTLIB_BACKEND",  matplotlib.get_backend())
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df1 = df.tail(max_length);
    if(name is not None):
        plt.switch_backend('Agg')
    fig, axs = plt.subplots(ncols=2, figsize=(32, 16))
    lColor = COMPONENT_COLOR;
    if(name is not None and name.endswith("Forecast")):
        lColor = FORECAST_COLOR;
    df1.plot.line(time, [signal, estimator, residue],
                  color=[SIGNAL_COLOR, lColor, RESIDUE_COLOR],
                  ax=axs[0] , grid = True, legend=False)
    add_patched_legend(axs[0] , [signal, estimator, residue])
    residues =  df1[residue].values

    import scipy.stats as scistats
    resid = residues[~np.isnan(residues)]
    scistats.probplot(resid, dist="norm", plot=axs[1])

    if(name is not None):
        plt.switch_backend('Agg')
        fig.savefig(name + '_decomp_output.' + format)
        plt.close(fig)

def decomp_plot_as_png_base64(df, time, signal, estimator, residue, name = None, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(residue in df.columns)

    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    df1 = df.tail(max_length);
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    lColor = COMPONENT_COLOR;
    if(name is not None and name.endswith("Forecast")):
        lColor = FORECAST_COLOR;
    df1.plot.line(time, [signal, estimator, residue],
                  color=[SIGNAL_COLOR, lColor, RESIDUE_COLOR],
                  ax=axs[0] , grid = True, legend = False)
    add_patched_legend(axs[0] , [signal, estimator, residue])
    residues =  df1[residue].values

    import scipy.stats as scistats
    resid = residues[~np.isnan(residues)]
    scistats.probplot(resid, dist="norm", plot=axs[1])

    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    plt.close(fig)
    return figdata_png.decode('utf8')
    

def prediction_interval_plot(df, time, signal, estimator, lower, upper, name = None, format='png', max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(lower in df.columns)
    assert(upper in df.columns)


    df1 = df.tail(max_length).copy();
    lMin = np.mean(df1[signal]) -  np.std(df1[signal]) * 3;
    lMax = np.mean(df1[signal]) +  np.std(df1[signal]) * 3;
    df1[lower] = df1[lower].apply(lambda x : x if (np.isnan(x) or x >= lMin) else np.nan);
    df1[upper] = df1[upper].apply(lambda x : x if (np.isnan(x) or x <= lMax) else np.nan);

    # last value of the signal
    lLastSignalPos = df1[signal].dropna().tail(1).index[0];
    lEstimtorValue = df1[estimator][lLastSignalPos];
    df1.loc[lLastSignalPos , lower] = lEstimtorValue;
    df1.loc[lLastSignalPos , upper] = lEstimtorValue;

    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if(name is not None):
        plt.switch_backend('Agg')
    fig, axs = plt.subplots(ncols=1, figsize=(16, 8))
    df1.plot.line(time, [signal, estimator, lower, upper],
                  color=[SIGNAL_COLOR, FORECAST_COLOR, LOWER_COLOR, UPPER_COLOR],
                  ax=axs, grid = True, legend=False)
    add_patched_legend(axs , [signal, estimator, lower, upper])

    x = df1[time];
    type1 = np.dtype(x)
    if(type1.kind == 'M'):
        x = x.apply(lambda t : t.date());
    axs.fill_between(x.values, df1[lower], df1[upper], color=SHADED_COLOR, alpha=.2)

    if(name is not None):
        plt.switch_backend('Agg')
        fig.savefig(name + '_prediction_intervals_output.' + format)
        plt.close(fig)
    

def prediction_interval_plot_as_png_base64(df, time, signal, estimator, lower, upper, name = None, max_length = 1000) :
    assert(df.shape[0] > 0)
    assert(df.shape[1] > 0)
    assert(time in df.columns)
    assert(signal in df.columns)
    assert(estimator in df.columns)
    assert(lower in df.columns)
    assert(upper in df.columns)


    df1 = df.tail(max_length).copy();
    lMin = np.mean(df1[signal]) -  np.std(df1[signal]) * 3;
    lMax = np.mean(df1[signal]) +  np.std(df1[signal]) * 3;
    df1[lower] = df1[lower].apply(lambda x : x if (np.isnan(x) or x >= lMin) else np.nan);
    df1[upper] = df1[upper].apply(lambda x : x if (np.isnan(x) or x <= lMax) else np.nan);

    # last value of the signal
    lLastSignalPos = df1[signal].dropna().tail(1).index;
    lEstimtorValue = df1[estimator][lLastSignalPos];
    df1.loc[lLastSignalPos , lower] = lEstimtorValue;
    df1.loc[lLastSignalPos , upper] = lEstimtorValue;

    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    fig, axs = plt.subplots(ncols=1, figsize=(16, 8))
    df1.plot.line(time, [signal, estimator, lower, upper],
                  color=[SIGNAL_COLOR, FORECAST_COLOR, FORECAST_COLOR, FORECAST_COLOR],
                  ax=axs, grid = True, legend=False)
    add_patched_legend(axs , [signal, estimator, lower, upper])

    x = df1[time];
    type1 = np.dtype(x)
    if(type1.kind == 'M'):
        x = x.apply(lambda t : t.date());
    axs.fill_between(x.values, df1[lower], df1[upper], color=SHADED_COLOR, alpha=.5)

    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    plt.close(fig)
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png.decode('utf8')


def qqplot_residues(df , residue):
    pass

def build_record_label(labels_list):
    out = "<f0>" + str(labels_list[0]);
    i = 1;
    for l in labels_list[1:]:
        out = out + " | <f" + str(i) + "> " + str(l) ;
        i = i + 1;
    return out + "";


def plot_hierarchy(structure , iAnnotations, name):
    import pydot
    graph = pydot.Dot(graph_type='graph', rankdir='LR', fontsize="12.0");
    graph.set_node_defaults(shape='record')
    lLevelsReversed = sorted(structure.keys(), reverse=True);
    for level in  lLevelsReversed:
        color = '#%02x%02x%02x' % (255, 255, 127 + int(128 * (1.0 - (level + 1.0) / len(lLevelsReversed))));
        for col in structure[level].keys():
            lLabel = col if iAnnotations is None else str(iAnnotations[col]);
            if iAnnotations is not None:
                lLabel = build_record_label(iAnnotations[col]);
            node_col = pydot.Node(col, label=lLabel, style="filled", fillcolor=color, fontsize="12.0")
            graph.add_node(node_col);
            for col1 in structure[level][col]:
                lLabel1 = col1
                if iAnnotations is not None:
                    lLabel1 = build_record_label(iAnnotations[col1]);
                color1 = '#%02x%02x%02x' % (255, 255, 128 + int(128 * (1.0 - (level + 2.0) / len(lLevelsReversed))));
                node_col1 = pydot.Node(col1, label=lLabel1, style="filled",
                                       fillcolor=color1, fontsize="12.0")
                graph.add_node(node_col1);
                lEdgeLabel = "";
                if iAnnotations is not None:
                    lEdgeLabel = iAnnotations[col + "_" + col1];
                lEdge = pydot.Edge(node_col, node_col1, color="red", label=lEdgeLabel, fontsize="12.0")
                graph.add_edge(lEdge)
    # print(graph.obj_dict)
    if(name is not None):
        graph.write_png(name);
    else:
        from IPython.display import Image, display
        plot1 = Image(graph.create_png())
        display(plot1)
