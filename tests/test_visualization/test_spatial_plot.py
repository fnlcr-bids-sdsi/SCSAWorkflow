import unittest
import anndata
import numpy as np
import matplotlib.pyplot as plt
from spac.visualization import spatial_plot
import itertools
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class SpatialPlotTestCase(unittest.TestCase):
    def setUp(self):
        # Set up test data
        num_rows = 100
        num_cols = 100

        # Generate data matrix
        features = np.random.randint(0, 100, size=(num_rows, num_cols))
        feature_names = [f'Intensity_{i}' for i in range(num_cols)]

        # Generate annotation metadata
        annotation_data = {
            'annotation1': np.random.choice(['A', 'B', 'C'], size=num_rows),
            'annotation2': np.random.normal(0, 1, size=num_rows),
            'annotation3': np.random.uniform(0, 1, size=num_rows)
        }

        # Create the AnnData object
        self.adata = anndata.AnnData(
            X=features,
            obs=annotation_data
        )

        numpy_array = np.random.uniform(0, 100, size=(num_rows, 2))
        self.adata.obsm["spatial"] = numpy_array
        self.adata.var_names = feature_names

        # Generate layer data
        layer_data = np.random.randint(0, 100, size=(num_rows, num_cols))
        self.adata.layers['Normalized'] = layer_data
        self.adata.layers['Standardized'] = layer_data

        self.spot_size = 10
        self.alpha = 0.5

    def test_invalid_adata(self):
        # Test when adata is not an instance of anndata.AnnData
        with self.assertRaises(ValueError):
            spatial_plot(None, self.spot_size, self.alpha)

    def test_invalid_layer(self):
        # Test when layer is not a string
        with self.assertRaises(ValueError):
            spatial_plot(self.adata, self.spot_size, self.alpha, layer=123)

    def test_invalid_feature(self):
        # Test when feature is not a string
        with self.assertRaises(ValueError):
            spatial_plot(self.adata, self.spot_size, self.alpha, feature=123)

    def test_invalid_annotation(self):
        # Test when annotation is not a string
        with self.assertRaises(ValueError):
            spatial_plot(
                self.adata,
                self.spot_size,
                self.alpha,
                annotation=123)

    def test_invalid_spot_size(self):
        # Test when spot_size is not an integer
        with self.assertRaises(ValueError):
            spatial_plot(self.adata, 10.5, self.alpha)

    def test_invalid_alpha(self):
        # Test when alpha is not a float
        with self.assertRaises(ValueError):
            spatial_plot(self.adata, self.spot_size, "0.5")

    def test_invalid_alpha_range(self):
        # Test when alpha is outside the range of 0 to 1
        with self.assertRaises(ValueError):
            spatial_plot(self.adata, self.spot_size, -0.5)

    def test_missing_annotation(self):
        # Test when annotation is None and feature is None
        with self.assertRaises(ValueError) as cm:
            spatial_plot(self.adata, self.spot_size, self.alpha)
        error_msg = str(cm.exception)
        err_msg_exp = "Both annotation and feature are None, " + \
            "please provide single input."
        self.assertEqual(error_msg, err_msg_exp)

    def test_invalid_annotation_name(self):
        # Test when annotation name is not found in the dataset
        with self.assertRaises(ValueError) as cm:
            spatial_plot(
                self.adata,
                self.spot_size,
                self.alpha,
                annotation='annotation4'
            )
        error_msg = str(cm.exception)
        err_msg_exp = 'The annotation "annotation4" not found in the dataset.' + \
                    ' Existing annotations are: annotation1, annotation2, annotation3'
        self.assertEqual(error_msg, err_msg_exp)


    def test_invalid_feature_name(self):
        # Test when feature name is not found in the layer
        with self.assertRaises(ValueError) as cm:
            spatial_plot(
                self.adata,
                self.spot_size,
                self.alpha,
                feature='feature1'
            )
        error_msg = str(cm.exception)
        err_msg_exp = "Feature feature1 not found," + \
            " please check the sample metadata."
        self.assertEqual(error_msg, err_msg_exp)

    def test_spatial_plot_annotation(self):
        # This test verifies that the spatial_plot function
        # correctly interacts with the spatial function,
        # and returns the expected Axes object when
        # the annotation parameter is used.

        # Mock the spatial function
        # This mock the spatial function to replace the
        # original implementation.The mock function, mock_spatial,
        # checks if the inputs match the expected values.
        # The purpose of mocking the spatial function with the
        # mock_spatial function is to simulate the behavior of the
        # original spatial function during testing and verify if the
        # inputs passed to it match the expected values.
        def mock_spatial(
                adata,
                layer,
                annotation,
                spot_size,
                alpha,
                vmin,
                vmax,
                ax,
                show,
                **kwargs):
            # Assert that the inputs match the expected values
            self.assertEqual(layer, None)
            self.assertEqual(annotation, 'annotation1')
            self.assertEqual(spot_size, self.spot_size)
            self.assertEqual(alpha, self.alpha)
            self.assertEqual(vmin, None)
            self.assertEqual(vmax, None)
            self.assertIsInstance(ax, plt.Axes)
            self.assertFalse(show)
            # Return a list containing the ax object to mimic
            # the behavior of the spatial function
            return [ax]

        # Mock the spatial function with the mock_spatial function
        spatial_plot.__globals__['sc.pl.spatial'] = mock_spatial

        # Create an instance of Axes
        ax = plt.Axes(
            plt.figure(),
            rect=[0, 0, 1, 1]
        )

        # Call the spatial_plot function with the ax object
        returned_ax_list = spatial_plot(
            adata=self.adata,
            spot_size=self.spot_size,
            alpha=self.alpha,
            annotation='annotation1',
            ax=ax
        )

        # Assert that the spatial_plot function returned a list
        # containing an Axes object with the same properties
        returned_ax = returned_ax_list[0]
        self.assertEqual(returned_ax.get_title(), ax.get_title())
        self.assertEqual(returned_ax.get_xlabel(), ax.get_xlabel())
        self.assertEqual(returned_ax.get_ylabel(), ax.get_ylabel())

        # Restore the original spatial function
        del spatial_plot.__globals__['sc.pl.spatial']

    def test_spatial_plot_feature(self):
        # Mock the spatial function
        def mock_spatial(
                adata,
                layer,
                feature,
                spot_size,
                alpha,
                vmin,
                vmax,
                ax,
                show,
                **kwargs):
            # Assert that the inputs match the expected values
            self.assertEqual(layer, None)  # Ensuring layer is None as expected
            self.assertEqual(feature, 'Intensity_10')  # Checking feature value
            self.assertEqual(spot_size, self.spot_size)  # Checking spot size
            self.assertEqual(alpha, self.alpha)  # Checking alpha
            self.assertEqual(vmin, 0)  # Checking vmin
            self.assertEqual(vmax, 100)  # Checking vmax
            self.assertIsInstance(ax, plt.Axes)  # Ensuring ax is an Axes object
            self.assertFalse(show)  # Ensuring show is False
            # Return a list containing the ax object to mimic the behavior of the spatial function
            return [ax]

        # Mock the spatial function with the mock_spatial function
        spatial_plot.__globals__['sc.pl.spatial'] = mock_spatial

        # Create an instance of Axes
        ax = plt.Axes(
            plt.figure(),
            rect=[0, 0, 1, 1]
        )

        # Call the spatial_plot function with the ax object and ensure layer is None
        returned_ax_list = spatial_plot(
            adata=self.adata,
            spot_size=self.spot_size,
            alpha=self.alpha,
            feature='Intensity_10',
            vmin=0,
            vmax=100,
            ax=ax,
            layer=None  # Explicitly passing None to layer
        )

        # Assert that the spatial_plot function returned a list containing an Axes object with the same properties
        returned_ax = returned_ax_list[0]
        self.assertEqual(returned_ax.get_title(), ax.get_title())
        self.assertEqual(returned_ax.get_xlabel(), ax.get_xlabel())
        self.assertEqual(returned_ax.get_ylabel(), ax.get_ylabel())

        # Restore the original spatial function
        del spatial_plot.__globals__['sc.pl.spatial']


    def test_spatial_plot_combos_feature(self):
        # Define the parameter combinations to test
        spot_sizes = [10, 20]
        alphas = [0.5, 0.8]
        vmins = [-999, 0, 5]
        vmaxs = [-999, 10, 20]
        features = ['Intensity_20', 'Intensity_80']
        layers = [None, 'Standardized']

        # Generate all combinations of parameters
        # excluding both None values for features and annotations

        parameter_combinations = list(itertools.product(
            spot_sizes, alphas, vmins, vmaxs, features, layers
        ))

        parameter_combinations = [
            params for params in parameter_combinations if not (
                params[4] is None and params[5] is None
            )
        ]

        for params in parameter_combinations:
            spot_size, alpha, vmin, vmax, feature, layer = params
            # Test the spatial_plot function with the
            # given parameter combination

            ax = spatial_plot(
                self.adata,
                spot_size,
                alpha,
                vmin=vmin,
                vmax=vmax,
                feature=feature,
                layer=layer
            )

            # Perform assertions on the spatial plot
            # Check if ax has data plotted
            self.assertTrue(ax[0].has_data())

    def test_spatial_plot_combos_annotation(self):
        # Define the parameter combinations to test
        spot_sizes = [10, 20]
        alphas = [0.5, 0.8]
        vmins = [-999, 0, 5]
        vmaxs = [-999, 10, 20]
        annotation = ['annotation1', 'annotation2']
        layers = [None, 'Normalized']

        # Generate all combinations of parameters
        # excluding both None values for features and annotations

        parameter_combinations = list(itertools.product(
            spot_sizes, alphas, vmins, vmaxs, annotation, layers
        ))

        parameter_combinations = [
            params for params in parameter_combinations if not (
                params[4] is None and params[5] is None
            )
        ]

        for params in parameter_combinations:
            spot_size, alpha, vmin, vmax, annotation, layer = params
            # Test the spatial_plot function with the
            # given parameter combination

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax = spatial_plot(
                self.adata,
                spot_size,
                alpha,
                vmin=vmin,
                vmax=vmax,
                annotation=annotation,
                layer=layer,
                ax=ax
            )

            plt.close(fig)
            # Perform assertions on the spatial plot
            # Check if ax has data plotted
            self.assertTrue(ax[0].has_data())


if __name__ == '__main__':
    unittest.main()
