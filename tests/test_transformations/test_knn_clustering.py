import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import knn_clustering


class TestKnnClustering(unittest.TestCase):
    def setUp(self):
        """
        Set up a test environment for KNN clustering.

        This method is run before each test in the TestKnnClustering class. It initializes a synthetic 
        AnnData object (`adata`) that simulates a dataset for supervised clustering tasks. The 
        dataset includes features and class annotations, with a portion of the labels intentionally 
        set to "no_label" to test the handling of missing values.

        The attributes of the created AnnData object include:

        - `adata.X`: A 2D numpy array representing the feature matrix, where:
            - The first half of the rows are generated from a normal distribution with a mean of 10.
            - The second half of the rows are generated from a normal distribution with a mean of 100.

        - `adata.obs['classes']`: Class annotations for each row in `adata`, where approximately 
          half of the rows have missing labels represented by "no_label". The labels are as follows:
            - 0: Corresponds to data points with a mean around 10.
            - 1: Corresponds to data points with a mean around 100.

        - `adata.obs['all_missing_classes']`: An array where all entries are set to "no_label", 
          simulating a scenario where no class labels are available.

        - `adata.obs['no_missing_classes']`: An array containing all class labels (0 and 1), 
          indicating that all data points have valid annotations.

        - `adata.obs['alt_classes']`: An alternative class label array where "no_label" entries 
          are replaced with NaN values, allowing for testing scenarios that require handling 
          missing values as NaNs.

        Additionally, this method sets up several attributes for use in tests:
        
        - `self.annotation`: A string representing the column name for class labels in `obs`.
        - `self.alt_annotation`: A string representing the column name for alternative class labels.
        - `self.layer`: A string indicating which layer of data to use for KNN clustering.
        - `self.features`: A list of feature names used in the AnnData object, which includes 
          "gene1" and "gene2".
        """
        #########
        # adata #
        #########

        # Generate 6 rows, two with mean centered at (10, 10) and two with means at (100, 100)
        data = np.array([
                    np.concatenate(
                            (
                                np.random.normal(10, 1, 3),
                                np.random.normal(10, 1, 3)
                            )
                        ),
                    np.concatenate(
                            (
                                np.random.normal(100, 1, 3),
                                np.random.normal(100, 1, 3)
                            )
                        ),
                ]).reshape(-1, 2)
 
        # Generate class labels, label 0 = mean at (10, 10), label 1 = mean at (100, 100)
        full_class_labels = np.array([0, 0, 0, 1, 1, 1],dtype=object)
        class_labels = np.array([0, 0, "no_label", "no_label", 1, 1],dtype=object)
        alt_class_labels = np.array([0, 0, np.nan, np.nan, 1, 1],dtype=object)
        
        # Wrap into an AnnData object
        self.dataset = data
        self.adata = AnnData(
            X=self.dataset, var=pd.DataFrame(index=["gene1", "gene2"])
        )

        self.adata.layers["counts"] = self.dataset
        self.adata.obsm["derived_features"] = self.dataset
        self.adata.obs["classes"] = class_labels

        # annotations with all labels missing or present
        self.adata.obs["all_missing_classes"] = np.array(["no_label" for x in full_class_labels])
        self.adata.obs["no_missing_classes"] = full_class_labels
        self.adata.obs["alt_classes"] = alt_class_labels

        # non-adata parameters for unittests
        self.annotation = "classes"
        self.alt_annotation = "alt_classes"
        self.layer = "counts"
        self.features = ["gene1", "gene2"]


    def test_typical_case(self):
        # This test checks if the function correctly adds 'knn' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'knn_features' in the AnnData object's uns attribute.
        knn_clustering(
            adata=self.adata,
            features=self.features,
            annotation=self.annotation,
            layer=self.layer,
            k = 2
        )
        self.assertIn("knn", self.adata.obs)
        self.assertEqual(self.adata.uns["knn_features"], self.features)

    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_annotation"
        # to the # AnnData object's obs attribute
        output_annotation_name = "my_output_annotation"
        knn_clustering(
            adata=self.adata,
            features=self.features,
            annotation=self.annotation,
            layer=self.layer,
            k = 2,
            output_annotation=output_annotation_name,
        )
        self.assertIn(output_annotation_name, self.adata.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        knn_clustering(
            adata=self.adata,
            features=self.features,
            annotation=self.annotation,
            layer=None,
            k = 2
        )
        self.assertIn("knn", self.adata.obs)
        self.assertEqual(self.adata.uns["knn_features"], self.features)

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer and checks the error message
        invalid_k_value = 'invalid'
        err_msg = (f"`k` must be a positive integer. Received value: `{invalid_k_value}`") 
        with self.assertRaisesRegex(ValueError, err_msg):
            knn_clustering(
                adata=self.adata,
                features=self.features,
                annotation=self.annotation,
                layer=self.layer,
                k=invalid_k_value,
            )

    def test_trivial_label(self):
        # This test checks if the data is fully labeled or missing labels for every datapoint
        # and the associated error messages 

        # all datapoints labeled
        no_missing_annotation = "no_missing_classes"
        err_msg = (f"All cells are labeled in the annotation `{no_missing_annotation}`. Please provide a mix of labeled and unlabeled data.") 
        with self.assertRaisesRegex(ValueError, err_msg):
            knn_clustering(
                adata=self.adata,
                features=self.features,
                annotation=no_missing_annotation,
                layer=self.layer,
                k = 2
            )

        # no datapoints labeled
        all_missing_annotation = "all_missing_classes"
        err_msg = (f"No cells are labeled in the annotation `{all_missing_annotation}`. Please provide a mix of labeled and unlabeled data.") 
        with self.assertRaisesRegex(ValueError, err_msg):
            knn_clustering(
                adata=self.adata,
                features=self.features,
                annotation="all_missing_classes",
                layer=self.layer,
                k = 2
            )

    def test_clustering_accuracy(self):
        knn_clustering(
            adata=self.adata,
            features=self.features,
            annotation=self.annotation,
            layer="counts",
            k=2,
        )

        self.assertIn("knn", self.adata.obs)
        self.assertEqual(len(np.unique(self.adata.obs["knn"])), 2)

    def test_associated_features(self):
        # Run knn using the derived feature and generate two clusters
        output_annotation = "derived_knn"
        associated_table = "derived_features"
        knn_clustering(
            adata=self.adata,
            features=None,
            annotation=self.annotation,
            layer=None,
            k=2,
            output_annotation=output_annotation,
            associated_table=associated_table,
        )

        self.assertEqual(len(np.unique(self.adata.obs[output_annotation])), 2)

    def test_missing_label(self):
        # This test checks that the missing label parameter works as intended
        #first knn call with normal data
        knn_clustering(
            adata=self.adata,
            features=self.features,
            annotation=self.annotation,
            layer="counts",
            k=2,
            output_annotation="knn_1",
            associated_table=None,
            missing_label = "no_label"
        )
        #second knn call with alt_class data
        knn_clustering(
            adata=self.adata,
            features=self.features,
            annotation=self.alt_annotation,
            layer="counts",
            k=2,
            output_annotation="knn_2",
            associated_table=None,
            missing_label = np.nan
        )

        #assert that they produce the same final label
        self.assertTrue(all(self.adata.obs["knn_1"]==self.adata.obs["knn_2"]))

if __name__ == "__main__":
    unittest.main()
