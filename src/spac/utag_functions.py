import scipy
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import typing as tp
import parmap # need to install parmap
import anndata
from tqdm import tqdm


def z_score(x):
    """
    Scale (divide by standard deviation) and center (subtract mean) 
    array-like objects.
    """
    return (x - x.min()) / (x.max() - x.min())

def sparse_matrix_dstack(
    matrices: tp.Sequence[scipy.sparse.csr_matrix],
) -> scipy.sparse.csr_matrix:
    """
    Diagonally stack sparse matrices.
    """

    n = sum([x.shape[0] for x in matrices])
    _res = list()
    i = 0
    for x in matrices:
        v = scipy.sparse.csr_matrix((x.shape[0], n))
        v[:, i : i + x.shape[0]] = x
        _res.append(v)
        i += x.shape[0]
    return scipy.sparse.vstack(_res)

def _parallel_message_pass(
    ad,
    radius: int,
    coord_type: str,
    set_diag: bool,
    mode: str,
):
    sq.gr.spatial_neighbors(ad, radius=radius, coord_type=coord_type, 
                            set_diag=set_diag)
    ad = custom_message_passing(ad, mode=mode)
    return ad


def custom_message_passing(adata, mode: str = "l1_norm"):
    # from scipy.linalg import sqrtm
    # import logging
    if mode == "l1_norm":
        A = adata.obsp["spatial_connectivities"]
        from sklearn.preprocessing import normalize
        affinity = normalize(A, axis=1, norm="l1")
    else:
        # Plain A_mod multiplication
        A = adata.obsp["spatial_connectivities"]
        affinity = A
    # logging.info(type(affinity))
    adata.X = affinity @ adata.X
    return adata


def low_variance_filter(adata):
    return adata[:, adata.var["std"] > adata.var["std"].median()]


def add_probabilities_to_centroid(
    adata, col: str, name_to_output: str = None
):
    from scipy.special import softmax

    if name_to_output is None:
        name_to_output = col + "_probabilities"

    mean = z_score(adata.to_df()).groupby(adata.obs[col]).mean()
    probs = softmax(adata.to_df() @ mean.T, axis=1)
    adata.obsm[name_to_output] = probs
    return adata

def utag(
    adata,
    channels_to_use = None,
    slide_key = "Slide",
    save_key: str = "UTAG Label",
    filter_by_variance: bool = False,
    max_dist: float = 20.0,
    normalization_mode: str = "l1_norm",
    keep_spatial_connectivity: bool = False,
    n_pcs = 10,
    apply_umap: bool = False,
    umap_kwargs: tp.Dict[str, tp.Any] = dict(),
    apply_clustering: bool = True,
    clustering_method: tp.Sequence[str] = ["leiden"],
    resolutions: tp.Sequence[float] = [0.05, 0.1, 0.3, 1.0],
    leiden_kwargs: tp.Dict[str, tp.Any] = None,
    parc_kwargs: tp.Dict[str, tp.Any] = None,
    parallel: bool = False,
    processes: int = 1,
    k=15,
    random_state=42,
):
    """
    Discover tissue architechture in single-cell imaging data
    by combining phenotypes and positional information of cells.

    Parameters
    ----------
    adata: AnnData
        AnnData object with spatial positioning of cells in obsm 'spatial' slot.
    channels_to_use: Optional[Sequence[str]]
        An optional sequence of strings used to subset variables to use.
        Default (None) is to use all variables.
    max_dist: float
        Maximum distance to cut edges within a graph.
        Should be adjusted depending on resolution of images.
        For imaging mass cytometry, where resolution is 1um, 20 often 
        gives good results. Default is 20.
    slide_key: {str, None}
        Key of adata.obs containing information on the batch structure 
        of the data. In general, for image data this will often be a variable 
        indicating the image so image-specific effects are removed from data.
        Default is "Slide".
    save_key: str
        Key to be added to adata object holding the UTAG clusters.
        Depending on the values of `clustering_method` and `resolutions`,
        the final keys will be of the form: {save_key}_{method}_{resolution}".
        Default is "UTAG Label".
    filter_by_variance: bool
        Whether to filter vairiables by variance.
        Default is False, which keeps all variables.
    max_dist: float
        Recommended values are between 20 to 50 depending on magnification.
        Default is 20.
    normalization_mode: str
        Method to normalize adjacency matrix.
        Default is "l1_norm", any other value will not use normalization.
    keep_spatial_connectivity: bool
        Whether to keep sparse matrices of spatial connectivity and distance 
        in the obsp attribute of the resulting anndata object. This could be 
        useful in downstream applications.
        Default is not to (False).
    n_pcs: Number of principal components to use for clustering. Default is 10.
    If None, no PCA is performed and clustering is done on features.
    apply_umap: bool
        Whether to build a UMAP representation after message passing.
        Default is False.
    umap_kwargs: Dict[str, Any]
        Keyword arguments to be passed to scanpy.tl.umap for dimensionality 
        reduction after message passing. Default is 10.0.
    apply_clustering: bool
        Whether to cluster the message passed matrix.
        Default is True.
    clustering_method: Sequence[str]
        Which clustering method(s) to use for clustering of the 
        message passed matrix. Default is ["leiden"].
    resolutions: Sequence[float]
        What resolutions should the methods in `clustering_method` be run at.
        Default is [0.05, 0.1, 0.3, 1.0].
    leiden_kwargs: dict[str, Any]
        Keyword arguments to pass to scanpy.tl.leiden.
    parc_kwargs: dict[str, Any]
        Keyword arguments to pass to parc.PARC.
    parallel: bool
        Whether to run message passing part of algorithm in parallel.
        Will accelerate the process but consume more memory.
        Default is True.
    processes: int
        Number of processes to use in parallel.
        Default is to use all available (-1).

    Returns
    -------
    adata: AnnData
        AnnData object with UTAG domain predictions for each cell in adata.obs,
        column `save_key`.
    """
    ad = adata.copy()

    if channels_to_use:
        ad = ad[:, channels_to_use]

    if filter_by_variance:
        ad = low_variance_filter(ad)

    if isinstance(clustering_method, list):
        clustering_method = [m.upper() for m in clustering_method]
    elif isinstance(clustering_method, str):
        clustering_method = [clustering_method.upper()]
    else:
        print(
            """Invalid Clustering Method. Clustering Method Should Either be 
            a string or a list"""
        )
        return
    assert all(m in ["LEIDEN", "KMEANS"] for m in clustering_method)

    if "KMEANS" in clustering_method:
        from sklearn.cluster import KMeans

    print("Applying UTAG Algorithm...")
    if slide_key:
        ads = [
            ad[ad.obs[slide_key] == slide].copy() for slide in ad.obs[slide_key].unique()
        ]
        ad_list = parmap.map(
            _parallel_message_pass,
            ads,
            radius=max_dist,
            coord_type="generic",
            set_diag=True,
            mode=normalization_mode,
            pm_pbar=True,
            pm_parallel=parallel,
            pm_processes=processes,
        )
        ad_result = anndata.concat(ad_list)
        if keep_spatial_connectivity:
            ad_result.obsp["spatial_connectivities"] = sparse_matrix_dstack(
                [x.obsp["spatial_connectivities"] for x in ad_list]
            )
            ad_result.obsp["spatial_distances"] = sparse_matrix_dstack(
                [x.obsp["spatial_distances"] for x in ad_list]
            )
    else:
        sq.gr.spatial_neighbors(ad, radius=max_dist, coord_type="generic", 
                                set_diag=True)
        ad_result = custom_message_passing(ad, mode=normalization_mode)
    
    if apply_clustering:
        if n_pcs == None:
            print("0 components")
            sc.pp.neighbors(ad_result, n_pcs=0, n_neighbors=k, random_state=random_state, use_rep="X")
        else:
            print(f"execute with {n_pcs} principal components")
            sc.tl.pca(ad_result, n_comps=n_pcs)
            sc.pp.neighbors(ad_result, n_neighbors=k, n_pcs=n_pcs, random_state=random_state, use_rep='X_pca')

        if apply_umap:
            print("Running UMAP on Input Dataset...")
            sc.tl.umap(ad_result, **umap_kwargs)

        for resolution in resolutions:

            res_key1 = save_key + "_leiden_" + str(resolution)
            res_key3 = save_key + "_kmeans_" + str(resolution)
            if "LEIDEN" in clustering_method:
                print(f"Applying Leiden Clustering at Resolution: {resolution}...")
                kwargs = dict()
                kwargs.update(leiden_kwargs or {})
                sc.tl.leiden(
                    ad_result, resolution=resolution, key_added=res_key1, **kwargs
                )
                add_probabilities_to_centroid(ad_result, res_key1)

            if "KMEANS" in clustering_method:
                print(f"Applying K-means Clustering at Resolution: {resolution}...")
                k = int(np.ceil(resolution * 10))
                kmeans = KMeans(n_clusters=k, random_state=1).fit(ad_result.obsm["X_pca"])
                ad_result.obs[res_key3] = pd.Categorical(kmeans.labels_.astype(str))
                add_probabilities_to_centroid(ad_result, res_key3)

    return ad_result