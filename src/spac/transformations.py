import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import warnings
import logging
import scanpy.external as sce
from spac.utils import check_table, check_annotation, check_feature
from scipy import stats
import umap as umap_lib
from scipy.sparse import issparse
from typing import List, Union, Optional
from numpy.lib import NumpyVersion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='SPAC:%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def phenograph_clustering(
        adata,
        features,
        layer=None,
        k=50,
        seed=None,
        output_annotation="phenograph",
        associated_table=None,
        **kwargs):
    """
    Calculate automatic phenotypes using phenograph.

    The function will add these two attributes to `adata`:
    `.obs["phenograph"]`
        The assigned int64 class by phenograph

    `.uns["phenograph_features"]`
        The features used to calculate the phenograph clusters

    Parameters
    ----------
    adata : anndata.AnnData
       The AnnData object.

    features : list of str
        The variables that would be included in creating the phenograph
        clusters.

    layer : str, optional
        The layer to be used in calculating the phengraph clusters.

    k : int, optional
        The number of nearest neighbor to be used in creating the graph.

    seed : int, optional
        Random seed for reproducibility.

    output_annotation : str, optional
        The name of the output layer where the clusters are stored.

    associated_table : str, optional
        If set, use the corresponding key `adata.obsm` to calcuate the
        Phenograph. Takes priority over the layer argument.


    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with the phenograph clusters
        stored in `adata.obs[output_annotation]`
    """

    _validate_transformation_inputs(
        adata=adata,
        layer=layer,
        associated_table=associated_table,
        features=features
    )

    if not isinstance(k, int) or k <= 0:
        raise ValueError("`k` must be a positive integer")

    if seed is not None:
        np.random.seed(seed)

    data = _select_input_features(
        adata=adata,
        layer=layer,
        associated_table=associated_table,
        features=features
    )

    phenograph_out = sce.tl.phenograph(data,
                                       clustering_algo="leiden",
                                       k=k,
                                       seed=seed,
                                       **kwargs)

    adata.obs[output_annotation] = pd.Categorical(phenograph_out[0])
    adata.uns["phenograph_features"] = features

def knn_clustering(
        adata,
        features,
        annotation,
        layer=None,
        k=50,
        output_annotation="knn",
        associated_table=None,
        missing_label = "no_label",
        **kwargs):
    """
    Calculate knn clusters using sklearn KNeighborsClassifier

    The function will add these two attributes to `adata`:
    `.obs[output_annotation]`
        The assigned int64 class labels by KNeighborsClassifier

    `.uns[output_annotation_features]`
        The features used to calculate the knn clusters

    Parameters
    ----------
    adata : anndata.AnnData
       The AnnData object.

    features : list of str
        The variables that would be included in fitting the KNN classifier.
    
    annotation : str
        The name of the annotation used for classifying the data

    layer : str, optional
        The layer to be used.

    k : int, optional
        The number of nearest neighbor to be used in creating the graph.

    output_annotation : str, optional
        The name of the output layer where the clusters are stored.

    associated_table : str, optional
        If set, use the corresponding key `adata.obsm` to calcuate the
        clustering. Takes priority over the layer argument.

    missing_label : str or int
        The value of missing annotations in adata.obs[annotation]

    Returns
    -------
    None
        adata is updated inplace 
    """

    # read in data, validate annotation in the call here
    _validate_transformation_inputs(
        adata=adata,
        layer=layer,
        associated_table=associated_table,
        features=features,
        annotation=annotation,
    )

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"`k` must be a positive integer. Received value: `{k}`")
    
    data = _select_input_features(
        adata=adata,
        layer=layer,
        associated_table=associated_table,
        features=features
    )

    # boolean masks for labeled and unlabeled data
    annotation_data = adata.obs[annotation]
    annotation_mask = annotation_data != missing_label
    annotation_mask &= pd.notnull(annotation_data)
    unlabeled_mask = ~annotation_mask

    # check that annotation is non-trivial
    if all(annotation_mask):
        raise ValueError(f"All cells are labeled in the annotation `{annotation}`. Please provide a mix of labeled and unlabeled data.")
    elif not any(annotation_mask):
        raise ValueError(f"No cells are labeled in the annotation `{annotation}`. Please provide a mix of labeled and unlabeled data.")

    # fit knn classifier to labeled data and predict on unlabeled data     
    data_labeled = data[annotation_mask]
    label_encoder = LabelEncoder()
    annotation_labeled = label_encoder.fit_transform(annotation_data[annotation_mask])
    
    classifier = KNeighborsClassifier(n_neighbors = k, **kwargs)
    classifier.fit(data_labeled, annotation_labeled)
    
    data_unlabeled = data[unlabeled_mask]
    knn_predict = classifier.predict(data_unlabeled)
    predicted_labels = label_encoder.inverse_transform(knn_predict)

    # format output and place predictions/data in right location
    adata.obs[output_annotation] = np.nan  
    adata.obs[output_annotation][unlabeled_mask] = predicted_labels
    adata.obs[output_annotation][annotation_mask] = annotation_data[annotation_mask]
    adata.uns[f"{output_annotation}_features"] = features

def get_cluster_info(adata, annotation, features=None, layer=None):
    """
    Retrieve information about clusters based on specific annotation.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    annotation : str
        Annotation in adata.obs for cluster info.
    features : list of str, optional
        Features (e.g., markers) for cluster metrics.
        Defaults to all features in adata.var_names.
    layer : str, optional
        The layer to be used in the aggregate summaries.
        If None, uses adata.X.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each cluster including the percentage of
        each cluster to the whole sample.
    """

    # Use utility functions for input validation
    check_annotation(adata, annotations=annotation)
    if features is None:
        features = list(adata.var_names)
    else:
        check_feature(adata, features=features)
    if layer:
        check_table(adata, tables=layer)

    # Convert data matrix or specified layer to DataFrame directly
    if layer:
        data_df = adata.to_df(layer=layer)
    else:
        data_df = adata.to_df()

    # Count cells in each cluster
    cluster_counts = adata.obs[annotation].value_counts().reset_index()
    cluster_counts.columns = [annotation, "Number of Cells"]

    # Calculate the percentage of cells in each cluster
    total_cells = adata.obs.shape[0]
    cluster_counts['Percentage'] = (
        cluster_counts['Number of Cells'] / total_cells
    ) * 100

    # Initialize DataFrame for cluster metrics
    cluster_metrics = pd.DataFrame({annotation: cluster_counts[annotation]})

    # Add cluster annotation
    data_df[annotation] = adata.obs[annotation].values

    # Calculate statistics for each feature in each cluster
    for feature in features:
        grouped = data_df.groupby(annotation)[feature]\
                            .agg(["mean", "median"])\
                            .reset_index()
        grouped.columns = [
            f"{col}_{feature}" if col != annotation else annotation
            for col in grouped.columns
        ]
        cluster_metrics = cluster_metrics.merge(
            grouped, on=annotation, how="left"
        )

    # Merge cluster counts and percentage
    cluster_metrics = pd.merge(
        cluster_metrics, cluster_counts,
        on=annotation, how="left"
    )

    return cluster_metrics


def tsne(adata, layer=None, **kwargs):
    """
    Perform t-SNE transformation on specific layer information.

    Parameters
    ----------
    adata : anndata.AnnData
       The AnnData object.
    layer : str
        Layer for phenograph cluster calculation.
    **kwargs
        Parameters for scanpy.tl.tsne function.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with t-SNE coordinates.
    """
    # If a layer is provided, it's transferred to 'obsm' for t-SNE computation
    # in scanpy.tl.tsne, which defaults to using the 'X' data matrix if not.

    if not isinstance(adata, anndata.AnnData):
        raise ValueError("adata must be an AnnData object.")

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")

        tsne_obsm_name = layer + "_tsne"
        X_tsne = adata.to_df(layer=layer)
        adata.obsm[tsne_obsm_name] = X_tsne
    else:
        tsne_obsm_name = None

    sc.tl.tsne(adata, use_rep=tsne_obsm_name, random_state=7)

    return adata


def run_umap(
        adata,
        n_neighbors=75,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=0,
        transform_seed=42,
        layer=None,
        output_derived_feature='X_umap',
        associated_table=None,
        **kwargs
):
    """
    Perform UMAP analysis on the specific layer of the AnnData object
    or the default data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_neighbors : int, default=75
        Number of neighbors to consider when constructing the UMAP. This
        influences the balance between preserving local and global structures
        in the data.
    min_dist : float, default=0.1
        Minimum distance between points in the UMAP space. Controls how
        tightly the embedding is allowed to compress points together.
    n_components : int, default=2
        Number of dimensions for embedding.
    metric : str, optional
        Metric to compute distances in high dimensional space.
        Check `https://umap-learn.readthedocs.io/en/latest/api.html` for
        options. The default is 'euclidean'.
    random_state : int, default=0
        Seed used by the random number generator(RNG) during UMAP fitting.
    transform_seed : int, default=42
        RNG seed during UMAP transformation.
    layer : str, optional
        Layer of AnnData object for UMAP. Defaults to `adata.X`.
    output_derived_feature : str, default='X_umap'
        The name of the column in adata.obsm that will contain the
        umap coordinates.
    associated_table : str, optional
        If set, use the corresponding key `adata.obsm` to calcuate the
        UMAP. Takes priority over the layer argument.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with UMAP coordinates stored in the `obsm`
        attribute. The key for the UMAP embedding in `obsm` is "X_umap" by
        default.
    """

    _validate_transformation_inputs(
        adata=adata,
        layer=layer,
        associated_table=associated_table
    )

    data = _select_input_features(
        adata=adata,
        layer=layer,
        associated_table=associated_table
    )

    # Convert data to pandas DataFrame for better memory handling
    data = pd.DataFrame(data.astype(np.float32))

    # Create and configure the UMAP model
    umap_model = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        low_memory=True,
        random_state=random_state,
        transform_seed=transform_seed,
        **kwargs
    )

    # Fit and transform the data with the UMAP model
    embedding = umap_model.fit_transform(data)

    # Store the UMAP coordinates back into the AnnData object under the
    # output_derived_feature key
    adata.obsm[output_derived_feature] = embedding

    return adata


def _validate_transformation_inputs(
        adata: anndata,
        layer: Optional[str] = None,
        associated_table: Optional[str] = None,
        features: Optional[Union[List[str], str]] = None,
        annotation: Optional[str] = None,
        ) -> None:
    """
    Validate inputs for transformation functions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Name of the layer in `adata` to use for transformation.
    associated_table : str, optional
        Name of the key in `obsm` that contains the numpy array.
    features : list of str or str, optional
        Names of features to use for transformation.
    annotation: str, optional
        Name of annotation column in `obs` that contains class labels

    Raises
    ------
    ValueError
        If both `associated_table` and `layer` are specified.
    """

    if associated_table is not None and layer is not None:
        raise ValueError("Cannot specify both"
                         f" 'associated table':'{associated_table}'"
                         f" and 'table':'{layer}'. Please choose one.")

    if associated_table is not None:
        check_table(adata=adata,
                    tables=associated_table,
                    should_exist=True,
                    associated_table=True)
    else:
        check_table(adata=adata,
                    tables=layer)

    if features is not None:
        check_feature(adata, features=features)
    
    if annotation is not None:
        check_annotation(adata, annotations=annotation)


def _select_input_features(adata: anndata,
                           layer: str = None,
                           associated_table: str = None,
                           features: Optional[Union[str, List[str]]] = None,

                           ) -> np.ndarray:
    """
    Selects the numpy array to be used as input for transformations

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer of AnnData object for UMAP. Defaults to `None`.
    associated_table : str, optional
        Name of the key in `adata.obsm` that contains the numpy array.
        Defaults to `None`.
    features : str or List[str], optional
        Names of the features to select from layer. If None, all features are
        selected. Defaults to None.

    Returns
    -------
    np.ndarray
        The selected numpy array.

    """
    if associated_table is not None:
        # Flatten the obsm numpy array before returning it
        logger.info(f'Using the associated table:"{associated_table}"')
        np_array = adata.obsm[associated_table]
        return np_array.reshape(np_array.shape[0], -1)
    else:
        np_array = adata.layers[layer] if layer is not None else adata.X
        logger.info(f'Using the table:"{layer}"')
        if features is not None:
            if isinstance(features, str):
                features = [features]

            logger.info(f'Using features:"{features}"')
            np_array = np_array[:,
                                [adata.var_names.get_loc(feature)
                                 for feature in features]]
        return np_array


def batch_normalize(adata, annotation, output_layer,
                    input_layer=None, method="median", log=False):
    """
    Adjust the features of every marker using a normalization method.

    The normalization methods are summarized here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8723144/

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    annotation: str
        The name of the annotation in `adata` to define batches.

    output_layer : str
        The name of the new layer to add to the anndata object for storing
        normalized data.

    input_layer : str, optional
        The name of the layer from which to read data. If None, read from `.X`.

    method : {"median", "Q50", "Q75", "z-score"}, default "median"
        The normalization method to use.

    log : bool, default False
        If True, take the log2 of features before normalization.
        Ensure this is boolean.
    """
    # Use utility functions for input validation
    check_annotation(adata, annotations=annotation)
    if input_layer:
        check_table(adata, tables=input_layer)

    if not isinstance(log, bool):
        logger.error("Argument 'log' must be of type bool.")
        raise ValueError("Argument 'log' must be of type bool.")

    allowed_methods = ["median", "Q50", "Q75", "z-score"]
    if method not in allowed_methods:
        raise ValueError(
            f"Unsupported normalization method '{method}', "
            f"allowed methods = {allowed_methods}"
        )

    # Create a copy of the input layer or '.X'
    if input_layer:
        original = pd.DataFrame(adata.layers[input_layer],
                                index=adata.obs.index).copy()
    else:
        original = pd.DataFrame(adata.X, index=adata.obs.index).copy()

    # Logarithmic transformation if required
    if log:
        original = np.log2(1+original)
        logger.info("Data transformed with log2")

    # Initialize the batch annotation values
    batches = adata.obs[annotation].unique().tolist()
    if method == "median" or method == "Q50":
        all_batch_quantile = original.quantile(q=0.5)
        logger.info("Median for al cells: %s", all_batch_quantile)
    elif method == "Q75":
        all_batch_quantile = original.quantile(q=0.75)
        logger.info("Q75 for all cells: %s", all_batch_quantile)
    elif method == "z-score":
        logger.info("Z-score setup is handled in batch processing loop.")

    # Normalize each batch
    for batch in batches:
        batch_cells = original[adata.obs[annotation] == batch]
        logger.info(f"Processing batch: {batch}, "
                    f"original values:\n{batch_cells}")

        if method == "median":
            batch_median = batch_cells.quantile(q=0.5)
            logger.info(f"Median for {batch}: %s", batch_median)
            original.loc[
                (adata.obs[annotation] == batch)
            ] = batch_cells + (all_batch_quantile - batch_median)

        elif method == "Q50":
            batch_50quantile = batch_cells.quantile(q=0.5)
            logger.info(f"Q50 for {batch}: %s", batch_50quantile)
            original.loc[adata.obs[annotation] == batch] = (
                batch_cells * all_batch_quantile / batch_50quantile
            )

        elif method == "Q75":
            batch_75quantile = batch_cells.quantile(q=0.75)
            logger.info(f"Q75 for {batch}: %s", batch_75quantile)
            original.loc[adata.obs[annotation] == batch] = (
                batch_cells * all_batch_quantile / batch_75quantile
            )

        elif method == "z-score":
            mean = batch_cells.mean()
            std = batch_cells.std(ddof=0)  # DataFrame.std() by default ddof=1
            logger.info(f"mean for {batch}: %s", mean)
            logger.info(f"std for {batch}: %s", std)
            # Ensure std is not zero by using a minimal threshold (e.g., 1e-8)
            std = np.maximum(std, 1e-8)
            original.loc[adata.obs[annotation] == batch] = \
                (batch_cells - mean) / std

    # Store normalized data in the specified output layer
    adata.layers[output_layer] = original
    logging.info(f"Normalization completed. Data in layer '{output_layer}'.")


def rename_annotations(adata, src_annotation, dest_annotation, mappings):
    """
    Rename labels in a given annotation in an AnnData object based on a
    provided dictionary. This function modifies the adata object in-place
    and creates a new annotation column.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    src_annotation : str
        Name of the column in adata.obs containing the original
        labels of the source annotation.
    dest_annotation : str
        The name of the new column to be created in the AnnData object
        containing the renamed labels.
    mappings : dict
        A dictionary mapping the original annotation labels to
        the new labels.

    Examples
    --------
    >>> adata = your_anndata_object
    >>> src_annotation = "phenograph"
    >>> mappings = {
    ...     "0": "group_8",
    ...     "1": "group_2",
    ...     "2": "group_6",
    ...     # ...
    ...     "37": "group_5",
    ... }
    >>> dest_annotation = "renamed_annotations"
    >>> adata = rename_annotations(
    ...     adata, src_annotation, dest_annotation, mappings)
    """

    # Use utility functions for input validation
    check_annotation(adata, annotations=src_annotation)

    # Inform the user about the data type of the original column
    original_dtype = adata.obs[src_annotation].dtype
    print(f"The data type of the original column '{src_annotation}' is "
          f"{original_dtype}.")

    # Convert the keys in mappings to the same data type as the unique values
    unique_values = adata.obs[src_annotation].unique()
    mappings = {
        type(unique_values[0])(key): value for key, value in mappings.items()
    }

    # Identify and handle unmapped labels
    missing_mappings = [
        key for key in unique_values if key not in mappings.keys()
    ]

    if missing_mappings:
        warnings.warn(
            f"Missing mappings for the following labels: {missing_mappings}. "
            f"They will be set to NaN in the '{dest_annotation}' column."
        )
        for missing in missing_mappings:
            mappings[missing] = np.nan  # Assign NaN for missing mappings

    # Create a new column based on the mappings
    adata.obs[dest_annotation] = (
        adata.obs[src_annotation].map(mappings).astype("category")
    )


def normalize_features(
    adata,
    low_quantile=0.02,
    high_quantile=0.98,
    interpolation="linear",
    input_layer=None,
    output_layer="normalized_feature",
    per_batch=False,
    annotation=None
):
    """
    Normalize the features stored in an AnnData object.
    Any entry lower than the value corresponding to low_quantile of the
    column will be assigned a value of low_quantile, and entry that
    are greater than high_quantile value will be assigned as the value of
    high_quantile.
    Other entries will be normalized with
    (values - quantile min)/(quantile max - quantile min).
    Resulting column will have value ranged between [0, 1].
    """
    # Check if the provided input_layer exists in the AnnData object
    if input_layer is not None:
        check_table(adata, tables=input_layer)

    # Check if output_layer already exists and issue a warning if it does
    if output_layer in adata.layers:
        warnings.warn(
            f"Layer '{output_layer}' already exists. It will be overwritten "
            "with the new transformed data."
        )

    if not isinstance(high_quantile, (int, float)):
        raise TypeError(
            "The high quantile should be a numeric value, currently got {}"
            .format(type(high_quantile))
        )
    if not isinstance(low_quantile, (int, float)):
        raise TypeError(
            "The low quantile should be a numeric value, currently got {}"
            .format(type(low_quantile))
        )
    if low_quantile >= high_quantile:
        raise ValueError(
            "The low quantile should be smaller than the high quantile, "
            "current values are: low quantile: {}, high_quantile: {}"
            .format(low_quantile, high_quantile)
        )
    if high_quantile <= 0 or high_quantile > 1:
        raise ValueError(
            "The high quantile value should be within (0, 1], current value:"
            " {}".format(high_quantile)
        )
    if low_quantile < 0 or low_quantile >= 1:
        raise ValueError(
            "The low quantile value should be within [0, 1), current value: {}"
            .format(low_quantile)
        )
    if interpolation not in ["nearest", "linear"]:
        raise ValueError(
            "interpolation must be either 'nearest' or 'linear', passed value"
            " is: {}".format(interpolation)
        )

    # Extract the data to transform
    if input_layer is not None:
        data_to_transform = adata.layers[input_layer]
    else:
        data_to_transform = adata.X

    # Ensure data is in numpy array format
    data_to_transform = (data_to_transform.toarray()
                         if issparse(data_to_transform)
                         else data_to_transform)

    if per_batch:
        if annotation is None:
            raise ValueError(
                "annotation must be provided if per_batch is True."
            )
        transformed_data = apply_per_batch(
            data_to_transform, adata.obs[annotation].values,
            method='normalize_features',
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            interpolation=interpolation
        )
    else:
        transformed_data = normalize_features_core(
            data_to_transform,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            interpolation=interpolation
        )

    # Store the transformed data in the specified output_layer
    adata.layers[output_layer] = pd.DataFrame(
        transformed_data,
        index=adata.obs_names,
        columns=adata.var_names
    )

    return adata


def normalize_features_core(data, low_quantile=0.02, high_quantile=0.98,
                            interpolation='linear'):
    """
    Normalize the features in a numpy array.

    Any entry lower than the value corresponding to low_quantile of the column
    will be assigned a value of low_quantile, and entries that are greater than
    high_quantile value will be assigned as value of high_quantile.
    Other entries will be normalized with
    (values - quantile min)/(quantile max - quantile min).
    Resulting column will have values ranged between [0, 1].

    Parameters
    ----------
    data : np.ndarray
        The data to be normalized.

    low_quantile : float, optional (default: 0.02)
        The lower quantile to use for normalization. Determines the
        minimum value after normalization.
        Must be a positive float between [0,1).

    high_quantile : float, optional (default: 0.98)
        The higher quantile to use for normalization. Determines the
        maximum value after normalization.
        Must be a positive float between (0,1].

    interpolation: str, optional (default: "linear")
        The interpolation method to use when selecting the value for
        low and high quantile. Values can be "nearest" or "linear".

    Returns
    -------
    np.ndarray
        The normalized data.

    Raises
    ------
    TypeError
        If low_quantile or high_quantile are not numeric.
    ValueError
        If low_quantile is not less than high_quantile, or if they are
        out of the range [0, 1] and (0, 1], respectively.
    ValueError
        If interpolation is not one of the allowed values.
    """
    if not isinstance(high_quantile, (int, float)):
        raise TypeError(
            "The high quantile should be a numeric value, "
            f"currently got {str(type(high_quantile))}")

    if not isinstance(low_quantile, (int, float)):
        raise TypeError(
            "The low quantile should be a numeric value, "
            f"currently got {str(type(low_quantile))}")

    if low_quantile < high_quantile:
        if high_quantile <= 0 or high_quantile > 1:
            raise ValueError(
                "The high quantile value should be within "
                f"(0, 1], current value: {high_quantile}")
        if low_quantile < 0 or low_quantile >= 1:
            raise ValueError(
                "The low quantile value should be within "
                f"[0, 1), current value: {low_quantile}")
    else:
        raise ValueError(
            "The low quantile should be smaller than "
            "the high quantile, current values are:\n"
            f"low quantile: {low_quantile}\n"
            f"high quantile: {high_quantile}")

    if interpolation not in ["nearest", "linear"]:
        raise ValueError(
            "Interpolation must be either 'nearest' or 'linear', "
            f"passed value is: {interpolation}")

    # Version check for numpy using NumpyVersion
    numpy_version = np.__version__
    if NumpyVersion(numpy_version) >= NumpyVersion('1.22.0'):
        # Use 'method' argument for newer versions
        quantiles = np.quantile(
            data, [low_quantile, high_quantile], axis=0,
            method=interpolation
        )
    else:
        # Use 'interpolation' argument for older versions
        quantiles = np.quantile(
            data, [low_quantile, high_quantile], axis=0,
            interpolation=interpolation
        )

    qmin = quantiles[0]
    qmax = quantiles[1]

    # Calculate range_values and identify zero ranges
    range_values = qmax - qmin
    zero_range_mask = range_values == 0

    # Prevent division by zero by setting zero ranges to 1 temporarily
    range_values = range_values.astype(float)
    range_values[zero_range_mask] = 1.0

    # Clip raw values to the quantile range before normalization
    clipped_data = np.clip(data, qmin, qmax)

    # Normalize the clipped values
    normalized_data = (clipped_data - qmin) / range_values

    # Correct normalized data for zero ranges
    normalized_data[:, zero_range_mask] = 0.0

    return normalized_data


def arcsinh_transformation(
    adata, input_layer=None, co_factor=None, percentile=None,
    output_layer="arcsinh", per_batch=False, annotation=None
):
    """
    Apply arcsinh transformation using a co-factor (fixed number) or a given
    percentile of each feature. This transformation can be applied to the
    entire dataset or per batch based on provided parameters.

    Computes the co-factor or percentile for each biomarker individually,
    ensuring proper scaling based on its unique range of expression levels.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to transform.
    input_layer : str, optional
        The name of the layer in the AnnData object to transform.
        If None, the main data matrix .X is used.
    co_factor : float, optional
        A fixed positive number to use as a co-factor for the transformation.
    percentile : float, optional
        The percentile is computed for each feature (column) individually.
    output_layer : str, default="arcsinh"
        Name of the layer to put the transformed results. If it already exists,
        it will be overwritten with a warning.
    per_batch : bool, optional, default=False
        Whether to apply the transformation per batch.
    annotation : str, optional
        The name of the annotation in `adata` to define batches. Required if
        per_batch is True.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object with the transformed data stored in the specified
        output_layer.
    """
    # Validate input parameters
    if co_factor is None and percentile is None:
        raise ValueError(
            "Both co_factor and percentile are None. Provide one to proceed."
        )

    if co_factor is not None and percentile is not None:
        raise ValueError(
            "Please specify either co_factor or percentile, not both."
        )

    if co_factor and co_factor <= 0:
        raise ValueError(
            f'Co_factor should be a positive value. Received: "{co_factor}"'
        )

    if percentile is not None and not (0 <= percentile <= 100):
        raise ValueError(
            f'Percentile should be between 0 and 100. Received: "{percentile}"'
        )

    # Check if the provided input_layer exists in the AnnData object
    if input_layer:
        check_table(adata, tables=input_layer)
        data_to_transform = adata.layers[input_layer]
    else:
        data_to_transform = adata.X

    # Ensure data is in numpy array format
    data_to_transform = data_to_transform.toarray() \
        if issparse(data_to_transform) else data_to_transform

    if per_batch:
        if annotation is None:
            raise ValueError(
                "annotation must be provided if per_batch is True."
            )
        check_annotation(
            adata, annotations=annotation, parameter_name="annotation"
        )
        transformed_data = apply_per_batch(
            data_to_transform, adata.obs[annotation].values,
            method='arcsinh_transformation', co_factor=co_factor,
            percentile=percentile
        )
    else:
        # Apply the arcsinh transformation using the core function
        transformed_data = arcsinh_transformation_core(
            data_to_transform, co_factor=co_factor, percentile=percentile
        )

    # Check if output_layer already exists and issue a warning if it does
    if output_layer in adata.layers:
        warnings.warn(
            f"Layer '{output_layer}' already exists. It will be overwritten "
            "with the new transformed data."
        )

    # Store the transformed data in the specified output_layer
    adata.layers[output_layer] = pd.DataFrame(
        transformed_data,
        index=adata.obs_names,
        columns=adata.var_names
    )

    return adata


def arcsinh_transformation_core(data, co_factor=None, percentile=None):
    """
    Apply arcsinh transformation using a co-factori or a percentile.

    Parameters
    ----------
    data : np.ndarray
        The data to transform.
    co_factor : float, optional
        A fixed positive number to use as a co-factor for the
        transformation.
    percentile : float, optional
        The percentile to determine the co-factor if co_factor is not
        provided.
        The percentile is computed for each feature (column)
        individually.

    Returns
    -------
    np.ndarray
        The transformed data.

    Raises
    ------
    ValueError
        If both co_factor and percentile are None.
        If both co_factor and percentile are specified.
        If percentile is not in the range [0, 100].
    """
    if co_factor is None and percentile is None:
        raise ValueError("Either co_factor or percentile must be provided.")

    if co_factor is not None and percentile is not None:
        raise ValueError(
            "Please specify either co_factor or percentile, not both.")

    if co_factor is None:
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile should be between 0 and 100.")
        co_factor = np.percentile(data, percentile, axis=0)

    # Perform arcsinh transformation with special handling for zero co_factor
    # If co_factor > 0, apply arcsinh(data / co_factor)
    # If co_factor == 0, apply arcsinh(data) to avoid division by zero
    transformed_data = np.where(co_factor > 0, np.arcsinh(data / co_factor),
                                np.arcsinh(data))

    return transformed_data


def z_score_normalization(adata, output_layer, input_layer=None, **kwargs):
    """
    Compute z-scores for the provided AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to normalize.
    output_layer : str
        The name of the layer to store the computed z-scores.
    input_layer : str, optional
        The name of the layer in the AnnData object to normalize.
        If None, the main data matrix .X is used.
    **kwargs : dict, optional
        Additional arguments to pass to scipy.stats.zscore.

    """

    # Check if the provided input_layer exists in the AnnData object
    if input_layer:
        check_table(adata, tables=input_layer)
        data_to_normalize = adata.layers[input_layer]
    else:
        data_to_normalize = adata.X

    # Ensure data is in numpy array format
    data_to_normalize = data_to_normalize.toarray() \
        if issparse(data_to_normalize) else data_to_normalize

    # Compute z-scores using scipy.stats.zscore
    normalized_data = stats.zscore(
        data_to_normalize, axis=0, nan_policy='omit', **kwargs
    )

    # Store the computed z-scores in the specified output_layer
    adata.layers[output_layer] = pd.DataFrame(
        normalized_data,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Print a message indicating that normalization is complete
    print(
        f'Z-score normalization completed. '
        f'Data stored in layer "{output_layer}".'
    )


def apply_per_batch(data, annotation, method, **kwargs):
    """
    Apply a given function to data per batch, with additional parameters.

    Parameters
    ----------
    data : np.ndarray
        The data to transform.
    annotation : np.ndarray
        Batch annotations for each row in the data.
    method : str
        The function to apply to each batch. Options: 'arcsinh_transformation'
        or 'normalize_features'.

    kwargs:
        Additional parameters to pass to the function.

    Returns
    -------
    np.ndarray
        The transformed data.
    """
    # Check data integrity
    if not isinstance(data, np.ndarray) or not isinstance(annotation,
                                                          np.ndarray):
        raise ValueError("data and annotation must be numpy arrays")
    if len(data) != len(annotation):
        raise ValueError("data and annotation must have the same number of "
                         "rows")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    method_dict = {
        'arcsinh_transformation': arcsinh_transformation_core,
        'normalize_features': normalize_features_core
    }

    if method not in method_dict:
        raise ValueError("method must be 'arcsinh_transformation' or "
                         "'normalize_features'")

    transform_function = method_dict[method]

    transformed_data = np.zeros_like(data)
    unique_batches = np.unique(annotation)

    for batch in unique_batches:
        batch_indices = np.where(annotation == batch)[0]
        batch_data = data[batch_indices]

        normalized_batch_data = transform_function(batch_data, **kwargs)

        transformed_data[batch_indices] = normalized_batch_data

    return transformed_data
