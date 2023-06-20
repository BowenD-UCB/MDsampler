from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Structure

from mdsampler.cgcnn import CGCNN


class Sampler(nn.Module):
    """Perform structure sampling on a given set of pymatgen.core.Structures with CGCNN
    This is useful when we're interested in selecting the
    most representative structures in an ensemble.
    """

    def __init__(self, model="CGCNN"):
        super().__init__()
        if model == "CGCNN":
            self.feature_gen = CGCNN()
        else:
            raise NotImplementedError

    def sample_structures(
        self, structures: list[Structure], num_out: int = 50, batch_size: int = 128
    ):
        """Perform sampling on list of structures using CGCNN featurizer
        Args:
            structures(List): list of structures
            num_out(int): number of output structures to be selected
            batch_size(int): batch size for CGCNN
        return:
            selected_structures(List)
            wcss(float):
                WCSS stands for "Within Cluster Sum of Squares".
                It's a measure used in k-means clustering and other similar algorithms
                to evaluate the quality of the clustering.
                The smaller means the better coverage of selected structures as a
                statistical representative for the original input distribution.
        """
        crystal_features = self.feature_gen.get_crystal_fea(
            structures, batch_size=batch_size
        )
        sampled_feas, sampled_indices, wcss = self.get_near_centroid_vectors(
            tensors=crystal_features,
            num_out=num_out,
        )
        print(f"wcss = {wcss}")
        print(f"returned the {sampled_indices} structures")
        return [structures[i] for i in sampled_indices], wcss

    def get_wcss_list(
        self,
        structures: list[Structure],
        test_num_outs: list = None,
        batch_size: int = 128,
    ):
        """Perform screening of WCSS on given set of outputs
        this is used to see if the num_out for the current input set of structures are
        representative enough for a healthy distribution
        Args:
            structures(List): list of structures
            test_num_outs(List): number of output structures to be selected
            batch_size(int): batch size for CGCNN
        return:
            test_num_outs(List): the num_out that has been tested
            wcss_list(List): the resulted wcss for each num_out.
        """
        crystal_features = self.feature_gen.get_crystal_fea(
            structures, batch_size=batch_size
        )
        if test_num_outs is None:
            test_num_outs = np.linspace(1, 100, 8, dtype=int)

        wcss_list = []
        for num_out in test_num_outs:
            sampled_feas, sampled_indices, wcss = self.get_near_centroid_vectors(
                tensors=crystal_features,
                num_out=num_out,
            )
            wcss_list.append(wcss)
        return test_num_outs, wcss_list

    def kmeans(self, x, num_clusters, distance="euclidean"):
        """K-means algorithm using PyTorch
        Args:
            x(tensor): tensor of shape [batch, feature_dim]
            num_clusters:
            distance:
        Returns:
            centroids, labels, wcss.
        """
        # Initialize centroids randomly from the data points
        perm = torch.randperm(x.size(0))
        centroids = x[perm[:num_clusters]]

        while True:
            # Compute distances from centroids to all data points
            if distance == "euclidean":
                distances = torch.norm(x[:, None] - centroids, dim=2)
            else:
                raise NotImplementedError

            # Assign each data point to the closest centroid
            labels = torch.argmin(distances, dim=1)

            # Compute new centroids as the mean of all data points assigned to them
            new_centroids = torch.stack(
                [x[labels == i].mean(0) for i in range(num_clusters)]
            )

            # If centroids haven't changed, we've converged
            if torch.all(new_centroids == centroids):
                break
            centroids = new_centroids

        # Compute WCSS
        wcss = torch.sum((x - centroids[labels]) ** 2)
        return centroids, labels, wcss

    def get_near_centroid_vectors(self, tensors: torch.Tensor, num_out: int):
        # Run K-means clustering
        centroids, labels, wcss = self.kmeans(tensors, num_out)

        # Select the most representative tensors
        near_centroid_tensors = []
        near_centroid_indices = []
        for i, centroid in enumerate(centroids):
            # Calculate distances from cluster center to all tensors
            cluster_indices = torch.where(labels == i)[0]
            cluster_tensors = tensors[cluster_indices]

            distances = torch.norm(cluster_tensors - centroid, dim=1)

            # Select the closest vector
            closest_idx_in_cluster = torch.argmin(distances)
            closest_vector = cluster_tensors[closest_idx_in_cluster]

            # Add to the list
            near_centroid_tensors.append(closest_vector)
            near_centroid_indices.append(cluster_indices[closest_idx_in_cluster].item())
        return near_centroid_tensors, near_centroid_indices, wcss.item()