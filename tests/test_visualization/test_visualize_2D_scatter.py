import unittest
import numpy as np
import pandas as pd
from spac.visualization import visualize_2D_scatter
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestVisualize2DScatter(unittest.TestCase):

    def setUp(self):
        """Prepare data for testing."""
        self.x = np.random.rand(10)
        self.y = np.random.rand(10)
        # Fixed categorical labels to ensure representation of each category
        fixed_labels = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
        self.labels_categorical = pd.Series(fixed_labels).astype('category')
        self.labels_continuous = np.random.rand(10)

    def test_invalid_input_type(self):
        """Test handling of invalid input types."""
        with self.assertRaises(ValueError) as context_manager:
            visualize_2D_scatter(1, self.y)
        self.assertEqual(str(context_manager.exception),
                         "x and y must be array-like.")

    def test_labels_length_mismatch(self):
        """Test handling of mismatched lengths between data and labels."""
        wrong_labels = ['A'] * 9  # Shorter than x and y
        with self.assertRaises(ValueError) as context_manager:
            visualize_2D_scatter(self.x, self.y, labels=wrong_labels)
        expected_msg = "Labels length should match x and y length."
        self.assertEqual(str(context_manager.exception), expected_msg)

    def test_invalid_theme(self):
        """Test handling of invalid themes."""
        with self.assertRaises(ValueError) as context_manager:
            visualize_2D_scatter(self.x, self.y, theme='invalid_theme')
        expected_msg = (
            "Theme 'invalid_theme' not recognized. Please use a valid theme."
        )
        self.assertEqual(str(context_manager.exception), expected_msg)

    def test_custom_point_size(self):
        """Test specifying a custom point size."""
        figure, axis = visualize_2D_scatter(self.x, self.y, point_size=100)
        self.assertEqual(axis.collections[0].get_sizes()[0], 100)

    def test_categorical_labels(self):
        """Test visualization with categorical labels."""
        figure, axis = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical
        )
        # Check if legend is present
        self.assertTrue(len(axis.get_legend().get_texts()) == 3)

    def test_annotate_cluster_centers(self):
        """Test annotation of cluster centers with categorical labels."""
        figure, axis = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical,
            annotate_centers=True
        )
        self.assertEqual(
            len(axis.texts), len(set(self.labels_categorical))
        )

    def test_legend_placement_categorical_labels(self):
        """Test placement of legend with categorical labels."""
        figure, axis = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical
        )

        # Ensure the legend exists
        legend = axis.get_legend()
        self.assertIsNotNone(legend, "Legend is not present.")

        # Get the bounding box of the legend in axes coordinates
        bbox = legend.get_bbox_to_anchor().transformed(
            axis.transAxes.inverted())

        # Check if the legend is placed outside the plot area on the right side
        self.assertTrue(bbox.x0 >= 1, "Legend is not placed outside the plot.")

        # Ensure the legend's labels match the expected labels
        expected_labels = sorted(['A', 'B', 'C'])
        legend_labels = sorted(text.get_text() for text in legend.get_texts())
        self.assertEqual(legend_labels, expected_labels)

    def test_continuous_labels(self):
        """Test visualization with continuous labels."""
        figure, axis = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_continuous
        )
        # Check if colorbar is present in the figure
        colorbar = figure.colorbar(axis.collections[0])
        self.assertIsNotNone(colorbar)

    def test_equal_aspect_ratio(self):
        """Test if the plot has an equal aspect ratio."""
        figure, axis = visualize_2D_scatter(self.x, self.y)
        aspect = axis.get_aspect()
        self.assertTrue(aspect == 'equal' or aspect == 1.0)

    def test_axis_titles(self):
        """Test if axis titles are set correctly."""
        x_axis_title = 'Test X Axis'
        y_axis_title = 'Test Y Axis'
        figure, axis = visualize_2D_scatter(
            self.x, self.y, x_axis_title=x_axis_title,
            y_axis_title=y_axis_title
        )
        self.assertEqual(axis.get_xlabel(), x_axis_title)
        self.assertEqual(axis.get_ylabel(), y_axis_title)

    def test_plot_title(self):
        """Test if plot title is set correctly."""
        plot_title = 'Test Plot Title'
        figure, axis = visualize_2D_scatter(
            self.x, self.y, plot_title=plot_title
        )
        self.assertEqual(axis.get_title(), plot_title)

    def test_color_representation(self):
        """Test if color representation description is included in legend."""
        color_representation = 'Test Color Representation'
        figure, axis = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical,
            color_representation=color_representation
        )
        legend = axis.get_legend()
        self.assertIn(color_representation, legend.get_title().get_text())


if __name__ == '__main__':
    unittest.main()
