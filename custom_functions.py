# IMPORTS
import pandas as pd
import numpy as np
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from keras import Model

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# MISC

def set_seed(my_seed=42):
    """"Sets Numpy/Random seeds so results are reproducible"""
    # Set numpy seed
    seed(my_seed)
    # Set random seed
    random.seed(my_seed)

def lgb_f1_score(preds, data):
    """
    Custom evaluation function to be used in LGBM models.
    Calculate the macro F1 score for given predictions and
    validation data.
    """
    labels = data.get_label()
    preds = preds.reshape(4, -1).T
    preds = preds.argmax(axis = 1)
    f_score = f1_score(labels , preds,  average = 'macro')
    return 'f1_score', f_score, True

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# PRE-PROCESSING FUNCTIONS
def calc_sparsity(column = pd.Series):
    """"Function to calculate the sparsity of a given column"""
    sparsity = 1.0 - np.count_nonzero(column) / float(column.size)
    return sparsity

def autoencode(lyric_tr, n_components):
    """Build, compile and fit an autoencoder for
    lyric data using Keras. Uses a batch normalised,
    undercomplete encoder with leaky ReLU activations.
    It will take a while to train.
    --------------------------------------------------
    lyric_tr = df of lyric training data
    n_components = int, number of output dimensions
    from encoder
    """
    n_inputs = lyric_tr.shape[1]
    # define encoder
    visible = Input(shape=(n_inputs,))

    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e) 
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    bottleneck = Dense(n_components)(e)

    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)

    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # train model
    model.fit(lyric_tr, lyric_tr, epochs=200,
                        batch_size=16, verbose=1, validation_split=0.2)
    
    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    
    # return the encoder only
    return encoder



def pre_process(train = pd.DataFrame,
                test = pd.DataFrame,
                reduction_method = "pca",
                n_components = 400):
    """
    Function to conduct pre-processing necessary for modelling.
    Categorical columns (mode and key) are converted to category type
    for easier modelling later. y/label data is converted from strings
    to numerics. Remaining feature data are normalised and lyric features
    are reduced in dimensions by either PCA, Truncated SVD or Keras Autoencoding.
    Normalisation and dim reduction are both fit on ONLY train data, since
    fitting on test data is bad practice.
    Returns train label, processed train feature data, processed test data
    and the label encoder to easily map submission values to genre strings.
    ----------------------------------------------------------------------
    reduction_method = str, "pca", "svd", "keras"- the method of dimensionality
    reduction used
    n_components = int, the number of dimensions you want the lyric data to be
    reduced to
    """
    # Separate into X and y
    y_train = train.playlist_genre
    y_test = test.playlist_genre
    X_train = train.drop(columns = "playlist_genre")
    X_test = test.drop(columns = "playlist_genre")
    
    # Make label into numeric
    # Convert from string to numerics
    label_encoder = LabelEncoder()
    label_train = label_encoder.fit_transform(y_train)
    label_test = label_encoder.transform(y_test)

    # Normalise both test and train data
    scaler = MinMaxScaler()
    # Fit on ONLY train data, bad practice to fit on test
    X_norm_tr = scaler.fit_transform(X_train)
    # Transform test data
    X_norm_te = scaler.transform(X_test)

    # Reconstruct dataframes
    X_norm_tr = pd.DataFrame(X_norm_tr, columns = X_train.columns)
    X_norm_te = pd.DataFrame(X_norm_te, columns = X_test.columns)

    # Convert mode and key back to categorical features
    X_norm_tr["audio_mode"] = X_train["audio_mode"].astype("category").reset_index(drop = True)
    X_norm_tr["audio_key"] = X_train["audio_key"].astype("category").reset_index(drop = True)
    X_norm_te["audio_mode"] = X_test["audio_mode"].astype("category").reset_index(drop = True)
    X_norm_te["audio_key"] = X_test["audio_key"].astype("category").reset_index(drop = True)
    
    # Get just lyric features
    lyric_tr = X_norm_tr.loc[:, "lyrics_aah":]
    lyric_te = X_norm_te.loc[:, "lyrics_aah":]

    if reduction_method == "pca":
        # Do principal component analysis / dimension reduction on sparse lyric features
        pca = PCA(n_components)
        # Fit on ONLY training data then transform
        reduced_tr = pd.DataFrame(pca.fit_transform(lyric_tr)).add_prefix("lyrics_pca_")
        # ONLY transform test data
        reduced_te = pd.DataFrame(pca.transform(lyric_te)).add_prefix("lyrics_pca_")

    if reduction_method == "svd":
        # Do truncated SVD dimension reduction on sparse lyric features
        svd = TruncatedSVD(n_components)
        # Fit on ONLY training data then transform
        reduced_tr = pd.DataFrame(svd.fit_transform(lyric_tr)).add_prefix("lyrics_svd_")
        # ONLY transform test data
        reduced_te = pd.DataFrame(svd.transform(lyric_te)).add_prefix("lyrics_svd_")
    
    # This will take a while
    if reduction_method == "keras":
        # Create and fit a Keras undercomplete encoder
        encoder = autoencode(lyric_tr, n_components) # Fit on only training data
        
        # Predict into reduced dimensions
        reduced_tr = pd.DataFrame(encoder.predict(lyric_tr)).add_prefix("lyrics_keras_")
        reduced_te = pd.DataFrame(encoder.predict(lyric_te)).add_prefix("lyrics_keras_")

        
        
    # Combine reduced dimension lyric features with audio features
    X_norm_tr = pd.concat([X_norm_tr.loc[:, :"audio_duration_ms"],
                          reduced_tr
                          ], axis = 1)

    # Combine reduced dimension lyric features with audio features
    X_norm_te = pd.concat([X_norm_te.loc[:, :"audio_duration_ms"],
                           reduced_te
                           ], axis = 1)


    return X_norm_tr, label_train, X_norm_te, label_test, label_encoder


# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# PLOTTING + FORMATTING

def plot_scree_pca(var_explained_ratio = list):
    """
    Plots a scree/elbow plot of proportion of variance explained
    vs the number of principal components as well as the cumulative
    proportion of variance explained.
    ---------------------------------------------------------------
    var_explained_ratio = array,  as returned by e.g.
        pca.explained_variance_ratio_
    """
    
    # Plot proportion of variance explained by principal components
    fig, ax = plt.subplots()
    # Reduce margins
    plt.margins(x=0.01)
    # Get cumuluative sum of variance explained
    cum_var_explained = np.cumsum(var_explained_ratio)

    # Plot cumulative sum
    ax.fill_between(range(len(cum_var_explained)), cum_var_explained,
                    alpha = 0.4, color = "tab:orange",
                    label = "Cum. Var.")
    ax.set_ylim(0, 1)

    # Plot actual proportions
    ax2 = ax.twinx()
    ax2.plot(range(len(var_explained_ratio)), var_explained_ratio,
             alpha = 1, color = "tab:blue", lw  = 4, ls = "--",
             label = "Var per PC")
    ax2.set_ylim(0, 0.005)

    # Add lines to indicate where good values of components may be
    ax.hlines(0.6, 0, var_explained_ratio.shape[0], color = "tab:green", lw = 3, alpha = 0.6, ls=":")
    ax.hlines(0.8, 0, var_explained_ratio.shape[0], color = "tab:green", lw = 3, alpha = 0.6, ls=":")

    # Plot both legends together
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    # Format axis as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1)) 

    # Add titles and labels
    ax.set_ylabel("Cum. Prop. of Variance Explained")
    ax2.set_ylabel("Prop. of Variance Explained per PC", rotation = 270, labelpad=30)
    ax.set_title("Variance Explained by Number of Principal Components")
    ax.set_xlabel("Number of Principal Components")


def plot_tsne(data):
    """
    Plots t-SNE values into 2D space for a range of
    principal component cut-offs, e.g. a facet grid
    for t-SNE with all PCs (1220) vs only 50 PCs
    """
    # Create grid
    g = sns.FacetGrid(data, col="Cutoff", hue = "y",
                    col_wrap = 2, height = 6,
                    palette=sns.color_palette("hls", 4),
                   # hue_kws=dict(alpha = 0.3)
                   )
    # Add plots
    g.map(sns.scatterplot, "tsne-2d-one", "tsne-2d-two", alpha = 0.3)
    # Add titles/legends
    g.fig.suptitle("t-SNE Plots vs Number of Principal Components Included", y = 1)
    g.add_legend()


def set_plot_config():
    """"Function to set-up Matplotlib plotting config
    for neater graphs"""
    plt.rcParams["figure.figsize"] = (17, 8)
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    plt.rc('font', **font)

