import re
import os
import pandas as pd
import anndata as ad
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from spac.utils import regex_search_list


def append_observation(
    data: pd.DataFrame,
    source_column: str,
    new_column: str,
    mapping_rules: list
) -> pd.DataFrame:
    """
    Append a new observation to a Pandas DataFrame based on mapping rules.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to which the new observation will be appended.
    source_column : str
        The name of the column in the 'data' DataFrame that
        contains the values to be mapped.
    new_column : str
        The name of the new column to be added to the DataFrame.
    mapping_rules : list
        A list of strings representing the mapping rules.
        Each string should have this format:
        <value in source column>:<value in new observation>

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new observation appended.
    """

    if source_column not in data.columns:
        error_msg = f"'{source_column}' does not exist in the DataFrame."
        raise ValueError(error_msg)

    if new_column in data.columns:
        error_msg = f"'{new_column}' already exist in the DataFrame."
        raise ValueError(error_msg)

    if not isinstance(mapping_rules, list):
        error_msg = "Mapping rules must be provided as a list."
        raise ValueError(error_msg)

    for rule in mapping_rules:
        if ':' not in rule:
            error_msg = f"Invalid mapping rule format: '{rule}'. " + \
                "It should have the format " + \
                "<value in new observation>:<value in source column>."
            raise ValueError(error_msg)

    # Create a copy of the input DataFrame to avoid modifying the original
    result_data = data.copy()

    # Initialize the new column with "Not_Mapped" for all rows
    result_data[new_column] = "Not_Mapped"

    # Store the original data type of the selected column
    original_dtype = result_data[source_column].dtype
    result_data[source_column] = result_data[source_column].astype(str)

    # Iterate over the mapping rules
    for rule in mapping_rules:
        # Split the rule into the source value and the new value
        source_value, new_value = rule.split(':')

        # Assign the new value to rows where
        # the source column matches the source value
        result_data.loc[
            result_data[source_column] == source_value, new_column
        ] = new_value

    # Restore the original data type of the selected column
    result_data[source_column] = result_data[
                                    source_column
                                ].astype(original_dtype)

    return result_data


def ingest_cells(dataframe,
                 regex_str,
                 x_col=None,
                 y_col=None,
                 annotation=None):

    """
    Read the csv file into an anndata object.

    The function will also intialize features and spatial coordiantes.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data frame that contains cells as rows, and cells informations as
        columns.

    regex_str : str or list of str
        A string or a list of strings representing python regular expression
        for the features columns in the data frame.  x_col : str The column
        name for the x coordinate of the cell.

    y_col : str
        The column name for the y coordinate of the cell.

    annotation : str or list of str
        The column name for the region that the cells. If a list is passed,
        multiple annotations will be created in the returned AnnData object.


    Returns
    -------
    anndata.AnnData
        The generated AnnData object
    """

    if not isinstance(regex_str, list):
        regex_list = [regex_str]
    else:
        regex_list = regex_str

    all_columns = list(dataframe.columns)
    all_features = []

    for column in regex_list:
        current_features = regex_search_list(
            [column],
            all_columns
        )

        if len(current_features) == 0:
            error_message = "Provided regex pattern(s) or feature(s):\n" + \
                f'"{column}"\n' + \
                "does not match any in the dataset, please review the input."
            raise ValueError(error_message)

        all_features.extend(current_features)

    features_df = dataframe[all_features]
    adata = ad.AnnData(
        features_df,
        dtype=features_df[all_features[0]].dtype)

    if annotation is not None:
        if isinstance(annotation, str):
            list_of_annotation = [annotation]
        else:
            list_of_annotation = annotation

        for annotation in list_of_annotation:

            # As selecting one column of the dataframe returns a series which
            # AnnData converts to NaN, then I convert it to a list before
            # assignment.
            adata.obs[annotation] = dataframe[annotation].tolist()

    if x_col is not None and y_col is not None:
        numpy_array = dataframe[[x_col, y_col]].to_numpy().astype('float32')
        adata.obsm["spatial"] = numpy_array
    return adata


def concatinate_regions(regions):
    """
    Concatinate data from multiple regions and create new indexes.

    Parameters
    ----------
    regions : list of anndata.AnnData
        AnnData objects to be concatinated.

    Returns
    -------
    anndata.AnnData
        New AnddData object with the concatinated values in AnnData.X

    """
    all_adata = ad.concat(regions)
    all_adata.obs_names_make_unique()
    return all_adata


def rescale_features(features, min_quantile=0.01, max_quantile=0.99):
    """
    Clip and rescale features outside the minimum and maximum quantile.

    The rescaled features will be between 0 and 1.

    Parameters
    ----------
    features : pandas.Dataframe
        The DataRrame of features.

    min_quantile : float
        The minimum quantile to be consider zero.

    max_quantile: float
        The maximum quantile to be considerd 1.

    Returns
    -------
    pandas.DataFrame
        The created DataFrame with normalized features.
    """
    markers_max_quantile = features.quantile(max_quantile)
    markers_min_quantile = features.quantile(min_quantile)

    features_clipped = features.clip(
        markers_min_quantile,
        markers_max_quantile,
        axis=1)

    scaler = MinMaxScaler()
    np_features_scaled = scaler.fit_transform(
        features_clipped.to_numpy())

    features_scaled = pd.DataFrame(
        np_features_scaled,
        columns=features_clipped.columns)

    return features_scaled


def add_rescaled_features(adata, min_quantile, max_quantile, layer):
    """
    Clip and rescale the features matrix.

    The results will be added into a new layer in the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    min_quantile : float
        The minimum quantile to rescale to zero.

    max_quantile : float
        The maximum quantile to rescale to one.

    layer : str
        The name of the new layer to add to the anndata object.
    """

    original = adata.to_df()
    rescaled = rescale_features(original, min_quantile, max_quantile)
    adata.layers[layer] = rescaled


def subtract_min_per_region(adata, annotation, layer, min_quantile=0.01):
    """
    Substract the minimum quantile of every marker per region.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    annotation: str
        The name of the annotation in `adata` to define batches.

    min_quantile : float
        The minimum quantile to rescale to zero.

    layer : str
        The name of the new layer to add to the AnnData object.
    """
    regions = adata.obs[annotation].unique().tolist()
    original = adata.to_df()

    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs[annotation] == region]
        new_features = subtract_min_quantile(region_cells, min_quantile)
        new_df_list.append(new_features)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df


def subtract_min_quantile(features, min_quantile=.01):
    """
    Subtract the features defined by the minimum quantile from all columns.

    Parameters
    ----------

    features : pandas.DataFrame
        The dataframe of features.

    min_quantile: float
        The minimum quantile to be consider zero.

    Returns
    -------
    pandas.DataFrame
        dataframe with rescaled features.
    """
    columns_min_quantile = features.quantile(min_quantile)

    subtracted_min = features - columns_min_quantile

    # Clip negative values to zero
    subtracted_min.clip(lower=0, axis=1, inplace=True)

    return subtracted_min


def load_csv_files(file_names):

    """
    Read the csv file(s) into a pandas dataframe.

    Parameters
    ----------
    file_names : str or list
        A list of csv file paths to be
        combined into single list of dataframe output

    Returns
    -------
    list
        A list of pandas dataframe of all the csv files.
    """

    meta_schema = []
    dataframe_list = []

    if not isinstance(file_names, list):
        if not isinstance(file_names, str):
            file_name_type = type(file_names)
            error_message = "file_names should be list or string" + \
                            ", but got " + str(file_name_type) + "."
            raise TypeError(error_message)
        else:
            file_names = [file_names]

    for file_name in file_names:

        # Check if the file exists
        if not os.path.exists(file_name):
            error_message = f"The file '{file_name}' does not exist."
            raise FileNotFoundError(error_message)

        # Try to load the csv into pandas DataFrame.
        # Check if the file exists
        if not os.path.exists(file_name):
            error_message = f"The file '{file_name}' does not exist."
            raise FileNotFoundError(error_message)

        # Check if the file is readable
        if not os.access(file_name, os.R_OK):
            error_message = "The file " + file_name + \
                    " cannot be read due to insufficient permissions."
            raise PermissionError(error_message)

        try:
            current_df = pd.read_csv(file_name)
        except pd.errors.EmptyDataError:
            error_message = "The file is empty or does not contain any data."
            raise TypeError(error_message)
        except pd.errors.ParserError:
            error_message = "The file could not be parsed. " + \
                            "Please check that the file is a valid CSV."
            raise TypeError(error_message)

        current_schema = current_df.columns.to_list()

        if len(meta_schema) == 0:
            meta_schema = current_schema
            print("Meta schema acquired. Columns are:")
            for column_name in meta_schema:
                print(column_name)

        if len(meta_schema) == len(current_schema):
            if set(meta_schema) != set(current_schema):
                error_message = "Column in current file does not match " + \
                        f"the meta_schema, got:\n {current_schema}. "
                raise ValueError(error_message)
        else:
            error_message = "Column in current file does not match " + \
                        f"the meta_schema, got:\n {current_schema}. "
            raise ValueError(error_message)

        dataframe_list.append([file_name, current_df])

    print("CSVs are converted into dataframes and combined into a list!")
    print("Total of " + str(len(dataframe_list)) + " dataframes in the list.")
    for each_file in dataframe_list:
        print("File name: ", each_file[0])
        print("Info: ")
        print(each_file[1].info())
        print("Description: ")
        print(each_file[1].describe())
        print()

    return dataframe_list


def combine_dfs(dataframes, annotations):

    """
    Combine a list of pandas dataframe into single pandas dataframe.

    Parameters
    ----------
    dataframes : list of tuple
        A list containing (file name, pandas dataframe) to be combined
        into single dataframe output

    annotations : pandas.DataFrame
        A pandas data frame where the index is the file name, and
        the columns are various annotations to
        add to all cells in a given dataframe.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame of all the cells
        where each cell has a unique index.
    """

    meta_schema = []
    combined_dataframe = pd.DataFrame()
    if not str(type(annotations)) == "<class 'pandas.core.frame.DataFrame'>":
        annotations_type = str(type(annotations))
        error_message = "annotations should be a pandas dataframe, " + \
            "but got " + annotations_type + "."
        raise TypeError(error_message)

    for current_df_list in dataframes:

        file_name = current_df_list[0]
        current_df = current_df_list[1]

        # Check is schema of each data_frame matches.
        # Check for length first, then check if columns match
        # The overall schema is based on the first file read.
        current_schema = current_df.columns.to_list()

        if len(meta_schema) == 0:
            meta_schema = current_schema
            print("Meta schema acquired. Columns are:")
            for column_name in meta_schema:
                print(column_name)

        if len(meta_schema) == len(current_schema):
            if set(meta_schema) != set(current_schema):
                error_message = "Column in current file does not match " + \
                        "the meta_schema, got:\n {current_schema}. "
                raise ValueError(error_message)
        else:
            error_message = "Column in current file does not match " + \
                        "the meta_schema, got:\n {current_schema}. "
            raise ValueError(error_message)

        # Check if the annotations DataFrame has the required index
        if file_name not in annotations.index:
            error_message = "Missing data in the annotations DataFrame" + \
                f"for the file '{file_name}'."
            raise ValueError(error_message)

        # Add annotations in to the dataframe
        file_annotations = annotations.loc[file_name]

        for file_annotation_name, file_annotation_value in \
                file_annotations.iteritems():
            current_df[file_annotation_name] = file_annotation_value

        if combined_dataframe.empty:
            combined_dataframe = current_df.copy()
        else:
            # Concatenate the DataFrames, with error handling
            try:
                combined_dataframe = pd.concat(
                    [combined_dataframe, current_df]
                    )
            except (ValueError, TypeError) as e:
                print('Error concatenating DataFrames:', e)

    # Reset index of the combined_dataframe
    combined_dataframe.reset_index(drop=True, inplace=True)

    print("CSVs are combined into single dataframe!")
    print(combined_dataframe.info())

    return combined_dataframe


def select_values(data, annotation, values=None):
    """
    Selects rows from input dataframe matching specified values in a column.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe.
    annotation : str
        The column name to be used for selection.
    values : list, optional
        List of values for annotation to include.
        If None, return all values.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing only the selected rows.

    Raises
    ------
    ValueError
        If annotation does not exist or one or more values passed
        do not exist in the specified column.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'column1': ['A', 'B', 'A', 'B', 'A'],
    ...     'column2': [1, 2, 3, 4, 5]
    ... })
    >>> select_values(df, 'column1', ['A'])
      column1  column2
    0       A        1
    2       A        3
    4       A        5
    """
    # Check if the DataFrame is empty
    if not data.empty:
        # If DataFrame is not empty, check if annotation exists
        if annotation not in data.columns:
            raise ValueError(
                f"Column {annotation} does not exist in the dataframe"
            )

        # If values exist in annotation, filter data
        if values is not None:
            filtered_data = data[data[annotation].isin(values)]
            if filtered_data.empty:
                warnings.warn("No matching values found in the data.")
            return filtered_data

    return data


def downsample_cells(data, annotation, n_samples=None,
                     stratify=False, rand=False):
    """
    Reduces the number of cells in the data by either selecting n_samples from
    every possible value of annotation, or returning n_samples
    stratified by the frequency of values in annotation.

    Parameters
    ----------
    data : pd.DataFrame
        The input data frame.
    annotation : str
        The column name to downsample on.
    n_samples : int, default=None
        The max number of samples to return for each group if stratify is
        False, or in total if stratify is True. If None, all samples returned.
    stratify : bool, default=False
        If true, stratify the returned values based on their input frequency.
    rand : bool, default=False
        If true and stratify is True, randomly select the returned cells.
        Otherwise, choose the first n cells.

    Returns
    -------
    data : pd.DataFrame
        The downsampled data frame.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...    'annotation': ['a', 'a', 'a', 'b', 'b', 'c'],
    ...    'value': [1, 2, 3, 4, 5, 6]
    ... })
    >>> print(downsample_cells(df, 'annotation', n_samples=2))
    """
    # Check if the column to downsample on exists
    if annotation not in data.columns:
        raise ValueError(
            f"Column {annotation} does not exist in the dataframe"
        )

    if n_samples is not None:
        # Stratify selection
        if stratify:
            # Determine frequencies of each group
            freqs = data[annotation].value_counts(normalize=True)
            n_samples_per_group = (freqs * n_samples).astype(int)
            samples = []
            # Group by annotation and sample from each group
            for group, group_data in data.groupby(annotation):
                n_group_samples = n_samples_per_group.get(group, 0)
                if rand:
                    # Randomly select the returned cells
                    samples.append(group_data.sample(min(n_group_samples,
                                                         len(group_data))))
                else:
                    # Choose the first n cells
                    samples.append(group_data.head(min(n_group_samples,
                                                       len(group_data))))
            # Concatenate all samples
            data = pd.concat(samples)
        else:
            # Non-stratified selection
            # Select the first n cells from each group
            data = data.groupby(annotation).apply(
                lambda x: x.head(n=min(n_samples, len(x)))
            ).reset_index(drop=True)

    # Print the number of rows in the resulting data
    print(f"Number of rows in the returned data: {len(data)}")

    return data


def calculate_centroid(
    data,
    x_min,
    x_max,
    y_min,
    y_max,
    new_x,
    new_y
):
    """
    Calculate the spatial coordinates of the cell centroid as the average of
    min and max coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        The input data frame. The dataframe should contain four columns for
        x_min, x_max, y_min, and y_max for centroid calculation.
    x_min : str
        column name with minimum x value
    x_max : str
        column name with maximum x value
    y_min : str
        column name with minimum y value
    y_max : str
        column name with maximum y value
    new_x : str
        the new column name of the x dimension of the centroid,
        allowing characters are alphabetic, digits and underscore
    new_y : str
        the new column name of the y dimension of the centroid,
        allowing characters are alphabetic, digits and underscore

    Returns
    -------
    data : pd.DataFrame
        dataframe with two new columns names

    """

    # Check for valid column names
    invalid_chars = r'[^a-zA-Z0-9_]'

    for name in [new_x, new_y]:
        if re.search(invalid_chars, name):
            error_string = "Column name " + str(name) + \
                " contains invalid characters. " + \
                "Use only alphanumeric characters and underscores."

            raise ValueError(error_string)

    # check if the columns exist in the dataframe
    for col in [x_min,
                x_max,
                y_min,
                y_max]:
        if col not in data.columns:
            raise ValueError(f"Column {col} does not exist in the dataframe.")

    # calculate the centroids
    data[new_x] = (data[x_min] + data[x_max]) / 2
    data[new_y] = (data[y_min] + data[y_max]) / 2

    return data


def bin2cat(data, one_hot_annotations, new_annotation):
    """
    Combine a set of columns representing
    a binary one hot encoding of categories
    into a new categorical column.

    Parameters:
    -----------
        data : pandas.DataFrame
            The pandas dataframe containing the one hot encoded annotations.

        one_hot_annotations : str or list of str
            A string or a list of strings representing
            python regular expression of the one hot encoded annotations
            columns in the data frame.

        new_annotation: str
            The column name for new categorical annotation to be created.

    Returns:
    --------
        pandas.DataFrame
            DataFrame with new categorical column added.

    Example:
    --------
    >>> data = pd.DataFrame({
    ...    'A': [1, 1, 0, 0],
    ...     'B': [0, 0, 1, 0]
    ... })
    >>> one_hot_annotations = ['A', 'B']
    >>> new_annotation = 'new_category'
    >>> result = bin2cat(data, one_hot_annotations, new_annotation)
    >>> print(result[new_annotation])
    0      A
    1      A
    2      B
    3    NaN
    Name: new_category, dtype: object
    """

    if isinstance(one_hot_annotations, str):
        one_hot_annotations = [one_hot_annotations]
    elif not isinstance(one_hot_annotations, list):
        error_string = "one_hot_annotations should " + \
                         "be a string or a list of strings."
        raise ValueError(error_string)

    if new_annotation in data.columns:
        raise ValueError("Column name for new annotation already exists.")

    if len(one_hot_annotations) > 0:
        # Add regrex to find cell labels

        all_columns = list(data.columns)
        all_cell_labels = regex_search_list(
                one_hot_annotations,
                all_columns
            )

        if len(all_cell_labels) > 0:
            cell_labels_df = data.loc[:, all_cell_labels]

            def get_columns_with_1(row):
                column_names = cell_labels_df.columns[row == 1]
                if len(column_names) > 1:
                    raise ValueError(f"Multiple instance found:{column_names}")
                elif len(column_names) == 1:
                    return column_names[0]
                else:
                    return np.nan

            column_names_with_1 = cell_labels_df.apply(
                get_columns_with_1,
                axis=1)
            column_names_with_1 = column_names_with_1.tolist()
            data[new_annotation] = column_names_with_1
            return data
        else:
            error_string = "No column was found in the dataframe " + \
                "with current regrex pattern(s)."
            raise ValueError(error_string)
