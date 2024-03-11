import unittest
from spac.visualization import generate_sankey_plot
from anndata import AnnData
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure


class TestGenerateSankeyPlot(unittest.TestCase):
    def setUp(self):
        # Create a simple AnnData object for testing
        self.adata = AnnData(
            np.random.rand(10, 10),
            obs=pd.DataFrame(
                {
                    'source_annotation': ['source1', 'source2'] * 5,
                    'target_annotation': ['target1', 'target2'] * 5,
                },
                index=[f'cell_{i}' for i in range(10)]
            )
        )

    def test_valid_type(self):
        # Test the function with valid inputs
        fig = generate_sankey_plot(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues',
            'Reds'
        )
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, Figure)

    def test_valid_labels(self):
        # Test the function with valid inputs
        fig = generate_sankey_plot(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues',
            'Reds'
        )

        expected_labels = [
            "Source_source1",
            "Source_source2",
            "Target_target1",
            "Target_target2"
        ]

        self.assertCountEqual(fig.data[0].node.label, expected_labels)

    def test_valid_link_setting(self):
        # Test the function with valid inputs
        fig = generate_sankey_plot(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues',
            'Reds'
        )

        # Check the source and target in fig.data[0].link
        expected_source = [0, 1]
        expected_target = [2, 3]
        expected_value = [5, 5]
        self.assertCountEqual(fig.data[0].link.source, expected_source)
        self.assertCountEqual(fig.data[0].link.target, expected_target)
        self.assertCountEqual(fig.data[0].link.value, expected_value)

    def test_invalid_source_annotation(self):
        # Test the function with an invalid source_annotation
        with self.assertRaises(ValueError):
            generate_sankey_plot(
                self.adata,
                'invalid_source_annotation',
                'target_annotation',
                'Blues',
                'Reds'
            )

    def test_invalid_target_annotation(self):
        # Test the function with an invalid target_annotation
        with self.assertRaises(ValueError):
            generate_sankey_plot(
                self.adata,
                'source_annotation',
                'invalid_target_annotation',
                'Blues',
                'Reds'
            )


if __name__ == '__main__':
    unittest.main()