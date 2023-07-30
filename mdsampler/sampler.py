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

    def __init__(self, model="CGCNN", use_device=None):
        super().__init__()
        self.device = use_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model == "CGCNN":
            self.feature_gen = CGCNN(use_device=self.device)
        else:
            raise NotImplementedError
        self.structures = []
        self.energies = []
        self.forces = []
        self.stresses = []
        self.magmoms = []
        self.features = None

    def add_structures(
        self,
        structures: list[Structure] | Structure,
        batch_size: int = 128,
        verbose: bool = None
    ):
        """Add structures to the sampler, and the sampler also stores their crystal feas
        Args:
            structures(List): list of structures
            batch_size(int): batch size for feature generator
        """
        if verbose is None:
            if len(structures) >= 100:
                verbose = True
            else:
                verbose = False
        crystal_features = self.feature_gen.get_crystal_fea(
            structures, batch_size=batch_size, verbose=verbose
        )
        if isinstance(structures, Structure):
            self.structures.append(structures)
        else:
            self.structures += structures
        if self.features is None:
            self.features = crystal_features
        else:
            self.features = torch.cat((self.features, crystal_features), dim=0)

    def add_data_points(
        self,
        structures: list[Structure] | Structure,
        energies: list,
        forces: list,
        stresses: list = None,
        magmoms: list = None,
        batch_size: int = 128,
        verbose: bool = None
    ):
        """Add structures to the sampler, and the sampler also stores their crystal feas
        Args:
            structures(List): list of structures
            energies(list): list of energy labels
            forces(list): list of force labels
            stresses(list): list of stress labels
            magmoms(list): list of magmom labels
            batch_size(int): batch size for feature generator
        """
        self.add_structures(structures=structures,
                            batch_size=batch_size,
                            verbose=verbose)
        self.energies += energies
        self.forces += forces
        if stresses is not None:
            self.stresses += stresses
        else:
            self.stresses += [None] * len(structures)
        if magmoms is not None:
            self.magmoms += magmoms
        else:
            self.magmoms += [None] * len(structures)

    def get_sampled_indices(
        self,
        num_out: int = 50,
        return_wcss=False
    ):
        """Perform sampling on list of structures using CGCNN featurizer
        Args:
            num_out(int): number of output structures to be selected

        return:
            selected_indices(List): the indices of the structures that are sampled
            wcss(float):
                WCSS stands for "Within Cluster Sum of Squares".
                It's a measure used in k-means clustering and other similar algorithms
                to evaluate the quality of the clustering.
                The smaller means the better coverage of selected structures as a
                statistical representative for the original input distribution.
        """
        sampled_feas, sampled_indices, wcss = self.get_near_centroid_vectors(
            tensors=self.features,
            num_out=num_out,
        )
        if return_wcss:
            return sampled_indices, wcss
        else:
            return sampled_indices

    def get_sampled_structures(
        self,
        num_out: int = 50,
        return_wcss=False,
        verbose=True,
    ):
        """Perform sampling on list of structures using CGCNN featurizer
        Args:
            num_out(int): number of output structures to be selected

        return:
            selected_structures(List)
            wcss(float):
                WCSS stands for "Within Cluster Sum of Squares".
                It's a measure used in k-means clustering and other similar algorithms
                to evaluate the quality of the clustering.
                The smaller means the better coverage of selected structures as a
                statistical representative for the original input distribution.
        """
        sampled_indices, wcss = self.get_sampled_indices(num_out=num_out,
                                                         return_wcss=True)
        if verbose:
            print(f"wcss = {wcss}")
            print(f"returned the {sampled_indices} structures")
        if return_wcss:
            return [self.structures[i] for i in sampled_indices], wcss
        else:
            return [self.structures[i] for i in sampled_indices]

    def save_sampled_data(
        self,
        num_out: int,
        save_name: str = 'sampled_data.p',
        targets: str = 'ef',
    ):
        """Saved sampled data using pickle
        Args:
            num_out(int): number of output structures to be selected
            save_name(str): name to save the file
            targets(str): training target
                Default = 'ef'

        return:
            None
        """
        import pickle
        sampled_indices = self.get_sampled_indices(num_out=num_out, return_wcss=False)
        if 's' in targets:
            sampled_stresses = [self.stresses[i] for i in sampled_indices]
        else:
            sampled_stresses = None
        if 'm' in targets:
            sampled_magmoms = [self.magmoms[i] for i in sampled_indices]
        else:
            sampled_magmoms = None
        saved_dataset = {
            'structures': [self.structures[i] for i in sampled_indices],
            'energies': [self.energies[i] for i in sampled_indices],
            'forces': [self.forces[i] for i in sampled_indices],
            'stresses': sampled_stresses,
            'magmoms': sampled_magmoms,
        }
        with open(save_name, 'wb') as file:
            pickle.dump(saved_dataset, file)
        print(f"Sampled file saved to {save_name}")
        return None

    def get_sampled_CHGNet_dataset(
        self,
        num_out: int,
        targets='ef',
    ):
        """Get sampled data in form of CHGNet dataset
        Args:
            num_out(int): number of output structures to be selected
            targets(str): training target
                Default = 'ef'

        return:
            CHGNet_dataset
        """
        from chgnet.data.dataset import StructureData
        sampled_indices = self.get_sampled_indices(num_out=num_out, return_wcss=False)
        if 's' in targets:
            sampled_stresses = [self.stresses[i] for i in sampled_indices]
        else:
            sampled_stresses = None
        if 'm' in targets:
            sampled_magmoms = [self.magmoms[i] for i in sampled_indices]
        else:
            sampled_magmoms = None
        dataset = StructureData(
            structures=[self.structures[i] for i in sampled_indices],
            energies=[self.energies[i] for i in sampled_indices],
            forces=[self.forces[i] for i in sampled_indices],
            stresses=sampled_stresses,
            magmoms=sampled_magmoms,
        )

        return dataset

    def get_wcss_list(
        self,
        test_num_outs: list = None,
    ):
        """Perform screening of WCSS on given set of outputs
        this is used to see if the num_out for the current input set of structures are
        representative enough for a healthy distribution
        Args:
            test_num_outs(List): List of number of output structures to be selected
        return:
            test_num_outs(List): the num_out that has been tested
            wcss_list(List): the resulted wcss for each num_out.
        """
        if test_num_outs is None:
            test_num_outs = np.linspace(1, 100, 8, dtype=int)

        wcss_list = []
        for num_out in test_num_outs:
            sampled_feas, sampled_indices, wcss = self.get_near_centroid_vectors(
                tensors=self.features,
                num_out=num_out,
            )
            wcss_list.append(wcss)
        return test_num_outs, wcss_list

    def kmeans(
        self,
        x,
        num_clusters: int,
        num_trial: int = 3,
        distance: str = "euclidean"
    ):
        """K-means algorithm using PyTorch
        Args:
            x(tensor): tensor of shape [batch, feature_dim]
            num_clusters:
            distance:
        Returns:
            centroids, labels
        """
        # Initialize centroids randomly from the data points
        perm = torch.randperm(x.size(0))
        centroids = x[perm[:num_clusters]]
        min_wcss = 999999
        best_centroids, best_labels = None, None
        for _ in range(num_trial):
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
                if wcss < min_wcss:
                    best_centroids = centroids
                    best_labels = labels
                    min_wcss = wcss

        return best_centroids, best_labels

    def get_near_centroid_vectors(
        self,
        tensors: torch.Tensor,
        num_out: int
    ):
        # Run K-means clustering
        centroids, labels = self.kmeans(tensors, num_out)

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

        # Calculate the distance to the near centroid tensors for WCSS
        near_centroid_tensors = torch.stack(near_centroid_tensors)
        wcss_near_centroid = torch.sum((tensors - near_centroid_tensors[labels]) ** 2)

        return near_centroid_tensors, near_centroid_indices, wcss_near_centroid.item()
