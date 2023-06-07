import seaborn as sns
import scanpy as sc
import seaborn
import pandas as pd
import numpy as np
import anndata
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def histogram(adata, feature_name=None, observation_name=None, layer=None,
              group_by=None, together=False, ax=None, **kwargs):
    """
    Plot the histogram of cells based on a specific feature from adata.X
    or observation from adata.obs.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.

    feature_name : str, optional
        Name of continuous feature from adata.X to plot its histogram.

    observation_name : str, optional
        Name of the observation from adata.obs to plot its histogram.

    group_by : str, default None
        Choose either to group the histogram by another column.

    together : bool, default False
        If True, and if group_by !=None create one plot for all groups.
        Otherwise, divide every histogram by the number of elements.

    ax : matplotlib.axes.Axes, optional
        An existing Axes object to draw the plot onto, optional.

    **kwargs
        Parameters passed to seaborn histplot function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes of the histogram plot.

    fig : matplotlib.figure.Figure
        The created figure for the plot.

    """
    # Validate inputs
    err = "adata must be an instance of anndata.AnnData, not {type(adata)}."
    if not isinstance(adata, anndata.AnnData):
        raise TypeError(err)

    df = adata.to_df()
    df = pd.concat([df, adata.obs], axis=1)

    if feature_name and observation_name:
        raise ValueError("Cannot pass both feature_name and observation_name,"
                         " choose one.")

    if feature_name:
        if feature_name not in df.columns:
            raise ValueError("feature_name not found in adata.")
        x = feature_name

    if observation_name:
        if observation_name not in df.columns:
            raise ValueError("observation_name not found in adata.")
        x = observation_name

    if group_by and group_by not in df.columns:
        raise ValueError("group_by not found in adata.")

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if group_by:
        groups = df[group_by].dropna().unique().tolist()
        n_groups = len(groups)
        if n_groups == 0:
            raise ValueError("There must be at least one group to create a"
                             " histogram.")
        if together:
            colors = sns.color_palette("hsv", n_groups)
            sns.histplot(data=df.dropna(), x=x, hue=group_by, multiple="stack",
                         palette=colors, ax=ax, **kwargs)
            return fig, ax
        else:
            fig, axs = plt.subplots(n_groups, 1, figsize=(5, 5*n_groups))
            if n_groups == 1:
                axs = [axs]
            for i, ax in enumerate(axs):
                sns.histplot(data=df[df[group_by] == groups[i]].dropna(),
                             x=x, ax=ax, **kwargs)
                ax.set_title(groups[i])
            return fig, axs

    sns.histplot(data=df, x=x, ax=ax, **kwargs)
    return fig, ax


def heatmap(adata, column, layer=None, **kwargs):
    """
    Plot the heatmap of the mean feature of cells that belong to a `column`.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    column : str
        Name of member of adata.obs to plot the histogram.

    layer : str, default None
        The name of the `adata` layer to use to calculate the mean feature.

    **kwargs:
        Parameters passed to seaborn heatmap function.

    Returns
    -------
    pandas.DataFrame
        A dataframe tha has the labels as indexes the mean feature for every
        marker.

    matplotlib.figure.Figure
        The figure of the heatmap.

    matplotlib.axes._subplots.AxesSubplot
        The AsxesSubplot of the heatmap.

    """
    features = adata.to_df(layer=layer)
    labels = adata.obs[column]
    grouped = pd.concat([features, labels], axis=1).groupby(column)
    mean_feature = grouped.mean()

    n_rows = len(mean_feature)
    n_cols = len(mean_feature.columns)
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    seaborn.heatmap(
        mean_feature,
        annot=True,
        cmap="Blues",
        square=True,
        ax=ax,
        fmt=".1f",
        cbar_kws=dict(use_gridspec=False, location="top"),
        linewidth=.5,
        annot_kws={"fontsize": 10},
        **kwargs)

    ax.tick_params(axis='both', labelsize=25)
    ax.set_ylabel(column, size=25)

    return mean_feature, fig, ax


def hierarchical_heatmap(adata, observation, layer=None, dendrogram=True,
                         standard_scale=None, ax=None, **kwargs):
    """
    Generates a hierarchical clustering heatmap.
    Cells are stratified by `observation`,
    then mean intensities are calculated for each feature across all cells
    to plot the heatmap using scanpy.tl.dendrogram and sc.pl.matrixplot.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    observation : str
        Name of the observation in adata.obs to group by and calculate mean
        intensity.
    layer : str, optional
        The name of the `adata` layer to use to calculate the mean intensity.
        Default is None.
    dendrogram : bool, optional
        If True, a dendrogram based on the hierarchical clustering between
        the `observation` categories is computed and plotted. Default is True.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new figure and axes
        object will be created. Default is None.
    **kwargs:
        Additional parameters passed to sc.pl.matrixplot function.

    Returns
    ----------
    mean_intensity : pandas.DataFrame
        A DataFrame containing the mean intensity of cells for each
        observation.
    matrixplot : scanpy.pl.matrixplot
        A Scanpy matrixplot object.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from spac.visualization import hierarchical_heatmap
    >>> import anndata

    >>> X = pd.DataFrame([[1, 2], [3, 4]], columns=['gene1', 'gene2'])
    >>> obs = pd.DataFrame(['type1', 'type2'], columns=['cell_type'])
    >>> all_data = anndata.AnnData(X=X, obs=obs)

    >>> fig, ax = plt.subplots()  # Create a new figure and axes object
    >>> mean_intensity, matrixplot = hierarchical_heatmap(all_data,
    ...                                                   "cell_type",
    ...                                                   layer=None,
    ...                                                   standard_scale='var',
    ...                                                   ax=None)
    # Display the figure
    # matrixplot.show()
    """

    # Check if observation exists in adata
    if observation not in adata.obs.columns:
        msg = (f"The observation '{observation}' does not exist in the "
               f"provided AnnData object. Available observations are: "
               f"{list(adata.obs.columns)}")
        raise KeyError(msg)

    # Check if the layer exists in adata
    if layer and layer not in adata.layers.keys():
        msg = (f"The layer '{layer}' does not exist in the "
               f"provided AnnData object. Available layers are: "
               f"{list(adata.layers.keys())}")
        raise KeyError(msg)

    # Raise an error if there are any NaN values in the observation column
    if adata.obs[observation].isna().any():
        raise ValueError("NaN values found in observation column.")

    # Calculate mean intensity
    intensities = adata.to_df(layer=layer)
    labels = adata.obs[observation]
    grouped = pd.concat([intensities, labels], axis=1).groupby(observation)
    mean_intensity = grouped.mean()

    # Reset the index of mean_feature
    mean_intensity = mean_intensity.reset_index()

    # Convert mean_intensity to AnnData
    mean_intensity_adata = sc.AnnData(
        X=mean_intensity.iloc[:, 1:].values,
        obs=pd.DataFrame(
            index=mean_intensity.index,
            data={
                observation: mean_intensity.iloc[:, 0]
                .astype('category').values
            }
        ),
        var=pd.DataFrame(index=mean_intensity.columns[1:])
    )

    # Compute dendrogram if needed
    if dendrogram:
        sc.tl.dendrogram(
            mean_intensity_adata,
            groupby=observation,
            var_names=mean_intensity_adata.var_names,
            n_pcs=None
        )

    # Create the matrix plot
    matrixplot = sc.pl.matrixplot(
        mean_intensity_adata,
        var_names=mean_intensity_adata.var_names,
        groupby=observation, use_raw=False,
        dendrogram=dendrogram,
        standard_scale=standard_scale, cmap="viridis",
        return_fig=True, ax=ax, show=False, **kwargs
    )
    return mean_intensity, matrixplot


def threshold_heatmap(adata, feature_cutoffs, observation):
    """
    Creates a heatmap for each feature, categorizing intensities into low,
    medium, and high based on provided cutoffs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the feature intensities in .X attribute.
    feature_cutoffs : dict
        Dictionary with feature names as keys and tuples with two intensity
        cutoffs as values.
    observation : str Column name in .obs DataFrame
        that contains the observation used for grouping.

    Returns
    -------
    Dictionary of :class:`~matplotlib.axes.Axes`
        A dictionary contains the axes of figures generated in the scanpy
        heatmap function.
        Consistent Key: 'heatmap_ax'
        Potential Keys includes: 'groupby_ax', 'dendrogram_ax', and
        'gene_groups_ax'.

    """

    # Assert observation is a string
    if not isinstance(observation, str):
        err_type = type(observation).__name__
        err_msg = (f'Observation should be string. Got {err_type}.')
        raise TypeError(err_msg)

    # Assert observation is a column in adata.obs DataFrame
    if observation not in adata.obs.columns:
        err_msg = f"'{observation}' not found in adata.obs DataFrame."
        raise ValueError(err_msg)

    if not isinstance(feature_cutoffs, dict):
        raise TypeError("feature_cutoffs should be a dictionary.")

    for key, value in feature_cutoffs.items():
        if not (isinstance(value, tuple) and len(value) == 2):
            raise ValueError(
                "Each value in feature_cutoffs should be a "
                "tuple of two elements."
            )
        if math.isnan(value[0]):
            raise ValueError(f"Low cutoff for {key} should not be NaN.")
        if math.isnan(value[1]):
            raise ValueError(f"High cutoff for {key} should not be NaN.")

    adata.uns['feature_cutoffs'] = feature_cutoffs

    intensity_df = pd.DataFrame(index=adata.obs_names,
                                columns=feature_cutoffs.keys())

    for feature, cutoffs in feature_cutoffs.items():
        low_cutoff, high_cutoff = cutoffs
        feature_values = adata[:, feature].X.flatten()
        intensity_df.loc[feature_values <= low_cutoff, feature] = 0
        intensity_df.loc[(feature_values > low_cutoff) &
                         (feature_values <= high_cutoff), feature] = 1
        intensity_df.loc[feature_values > high_cutoff, feature] = 2

    intensity_df = intensity_df.astype(int)
    adata.layers["intensity"] = intensity_df.to_numpy()
    adata.obs[observation] = adata.obs[observation].astype('category')

    color_map = {0: (0/255, 0/255, 139/255), 1: 'green', 2: 'yellow'}
    colors = [color_map[i] for i in range(3)]
    cmap = ListedColormap(colors)

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    heatmap_plot = sc.pl.heatmap(
        adata,
        var_names=intensity_df.columns,
        groupby=observation,
        use_raw=False,
        layer='intensity',
        cmap=cmap,
        norm=norm,
        swap_axes=True,
        show=False
    )

    colorbar = plt.gcf().axes[-1]
    colorbar.set_yticks([0, 1, 2])
    colorbar.set_yticklabels(['low', 'medium', 'high'])

    return heatmap_plot
