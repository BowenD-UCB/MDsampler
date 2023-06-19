"""CGCNN Wrapper."""
from __future__ import annotations

import argparse
import os
import shutil
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Structure
from torch import Tensor
from torch.autograd import Variable

from mdsampler.cgcnn.data import (
    AtomCustomJSONInitializer,
    CIFData,
    GaussianDistance,
    collate_pool,
    get_train_val_test_loader,
)
from mdsampler.cgcnn.model import CrystalGraphConvNet

pjoin = os.path.join
module_dir = os.path.dirname(__file__)
model_dir = pjoin(module_dir, "pre-trained", "formation-energy-per-atom.pth.tar")


class CGCNN(nn.Module):
    """Wrapper to generate cgcnn energy prediction model."""

    def __init__(
        self,
        model_path: str = model_dir,
        orig_atom_fea_len: int = 92,
        nbr_fea_len: int = 41,
        max_num_nbr: int = 12,
        radius: float = 8,
        dmin: float = 0,
        step: float = 0.2,
    ):
        """Init CGCNN.

        Args:
            model_path(str): path of model
            orig_atom_fea_len(int): Number of atom features in the input.
                                    i.e. Original atom feature length
                                    (default 92)
            nbr_fea_len(int): Number of bond features.
                            i.e. Number of neighbors (default 41).

        """
        super().__init__()
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model_args = argparse.Namespace(**checkpoint["args"])

        self.input_generator = CGCNNInput(
            max_num_nbr=max_num_nbr, radius=radius, dmin=dmin, step=step
        )
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_h=self.model_args.n_h,
            n_conv=self.model_args.n_conv,
            h_fea_len=self.model_args.h_fea_len,
        )
        self.normalizer = CGCNNNormalizer(torch.zeros(3))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.normalizer.load_state_dict(checkpoint["normalizer"])

    def predict_energy(self, structure: Structure) -> np.ndarray:  # type: ignore
        """CGCNN predict formatio nenergy from pymatgen structure.

        Args:
            structure(Structure): structure to be predicted

        Returns: formation energy (eV/atom) of provided structure

        """
        self.model.eval()
        inp = self.input_generator.generate_input(structure)
        inp = (*inp, [torch.LongTensor(np.arange(structure.num_sites))])
        print(inp)
        output = self.model.get_prediction(*inp)
        return self.normalizer.denorm(output).data.cpu().numpy()[0][0]

    def get_crystal_fea(
        self,
        structure: Structure | list[Structure],
        batch_size: int = 128,
    ) -> Tensor:
        """CGCNN predict formatio nenergy from pymatgen structure.

        Args:
            structure(Structure): structure to be predicted
            batch_size(int): batch size for CGCNN prediction
        Returns: formation energy (eV/atom) of provided structure

        """
        self.model.eval()
        if isinstance(structure, Structure):
            inp = self.input_generator.generate_input(structure)
            inp = (*inp, [torch.LongTensor(np.arange(structure.num_sites))])
            output = self.model(*inp)
            return output.detach().cpu()

        outputs = []
        for i in range(0, len(structure), batch_size):
            batch_structures = structure[i : i + batch_size]
            inp = self.input_generator.generate_inputs(batch_structures)
            outputs.append(self.model(*inp).detach().cpu())
        return torch.cat(outputs)


class CGCNNInput:
    """Wrapper to generate input for cgcnn from pymatgen structure."""

    atom_init_filename = pjoin(module_dir, "pre-trained", "atom_init.json")

    def __init__(
        self,
        max_num_nbr: int = 12,
        radius: float = 8,
        dmin: float = 0,
        step: float = 0.2,
        random_seed=123,
    ):
        """Init CGCNNInput.

        Args:
            max_num_nbr(int): The maximum number of neighbors
                            while constructing the crystal graph
                            (default 12)
            radius(float): The cutoff radius for searching neighbors
                            (default 8)
            dmin(float): The minimum distance for constructing
                        GaussianDistance (default 0)
            step(float): The step size for constructing GaussianDistance
                        (default 0.2)
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.random_seed = random_seed
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.ari = AtomCustomJSONInitializer(self.atom_init_filename)

    def _get_nbr_fea(self, all_nbrs: list, cif_id: int) -> tuple[np.ndarray, ...]:
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id)
                )
                nbr_fea_idx.append(
                    [x[2] for x in nbr] + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    [x[1] for x in nbr]
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append([x[2] for x in nbr[: self.max_num_nbr]])
                nbr_fea.append([x[1] for x in nbr[: self.max_num_nbr]])
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)  # type: ignore
        nbr_fea = self.gdf.expand(nbr_fea)
        return (nbr_fea_idx, nbr_fea)  # type: ignore

    def generate_input(
        self, structure: Structure, cif_id: int = None
    ) -> tuple[Any, ...]:
        """Generate cgcnn inputs for given structure.

        Args:
            structure(Structure): structure to get input for
            cif_id(int): Optional, the id of the structure

        Returns: Tuple of input (atom_fea, nbr_fea, nbr_fea_idx)
        """
        atom_fea = [
            sum([(self.ari.get_atom_fea(el.Z) * oc) for el, oc in site.species.items()])
            for site in structure
        ]
        atom_fea = np.vstack(atom_fea)  # type: ignore
        atom_fea = Tensor(atom_fea)  # type: ignore
        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        # sort the nbrs by distance
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        if cif_id:
            nbr_fea_idx, nbr_fea = self._get_nbr_fea(all_nbrs, cif_id)
        else:
            nbr_fea_idx, nbr_fea = self._get_nbr_fea(all_nbrs, 0)
        nbr_fea = Tensor(nbr_fea)  # type: ignore
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)  # type: ignore
        return (atom_fea, nbr_fea, nbr_fea_idx)

    def generate_inputs(self, structures: list[Structure], cif_ids: list[int] = None):
        """Generate cgcnn inputs for given list of structures
        Args:
            structures (list): List of structures to get inputs for.
            cif_ids (list): Optional, the list of ids of the structures.

        """
        if not cif_ids:
            cif_ids = list(range(len(structures)))
        batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
        crystal_atom_idx = []
        base_idx = 0
        for structure, cif_id in zip(structures, cif_ids):
            atom_fea, nbr_fea, nbr_fea_idx = self.generate_input(structure, cif_id)
            n_i = atom_fea.shape[0]  # number of atoms for this crystal
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
            new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
            crystal_atom_idx.append(new_idx)
            base_idx += n_i
        return (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        )

    def train_val_loader(self, data_path, batch_size=16, train_ratio=0.9):
        dataset = CIFData(
            data_path,
            self.max_num_nbr,
            self.radius,
            self.dmin,
            self.step,
            self.random_seed,
        )
        collate_fn = collate_pool
        train_loader, val_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=1,
            val_ratio=1 - train_ratio,
            test_ratio=0,
            pin_memory=False,
            train_size=None,
            val_size=None,
            test_size=None,
            return_test=False,
        )
        print(
            "CIFs imported: {} [train:{} ;val:{}]*{}".format(
                len(dataset), len(train_loader), len(val_loader), batch_size
            )
        )
        return train_loader, val_loader


class CGCNNNormalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor: Tensor):
        """Tensor is taken as a sample to calculate the mean and std.

        Args:
            tensor(Tensor): data
        """
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor: Tensor) -> Tensor:
        """Normalize tensor.

        Args:
            tensor(Tensor): data

        Returns: normalized tensor

        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: Tensor) -> Tensor:
        """Denormalize tensor.

        Args:
            normed_tensor(Tensor): normalized tensor data

        Returns: denormalized tensor

        """
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict:
        """Get dict of mean and std.

        Returns: dict of mean and std of the normalizer。

        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the normalizer with mean and std.

        Args:
            state_dict(Dict): dict of mean and std

        Returns: None

        """
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def train(train_loader, model, criterion, optimizer, epoch, normalizer, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_var = (
                Variable(input[0].cuda(non_blocking=True)),
                Variable(input[1].cuda(non_blocking=True)),
                input[2].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
            )
        else:
            input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        # normalize target
        target_normed = normalizer.norm(target)

        if torch.cuda.is_available():
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    mae_errors=mae_errors,
                )
            )


def validate(
    val_loader, model, criterion, normalizer, test=False, savepath=".", print_freq=10
):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        target_normed = normalizer.norm(target)

        if torch.cuda.is_available():
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    mae_errors=mae_errors,
                )
            )

    if test:
        star_label = "**"
        import csv

        save_csv = os.path.join(savepath, "training_results.csv")
        with open(save_csv, "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = "*"
    print(
        " {star} MAE {mae_errors.avg:.4f}".format(
            star=star_label, mae_errors=mae_errors
        )
    )
    return mae_errors.avg


def mae(prediction, target):
    """Computes the mean absolute error between prediction and target.

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", savepath="."):
    save_name = os.path.join(savepath, filename)
    best_save_name = os.path.join(savepath, "model_best.pth.tar")
    torch.save(state, save_name)
    if is_best:
        shutil.copyfile(save_name, best_save_name)


def expected_radius(struct):
    element_list = struct.composition.chemical_system.split("-")
    element_list = [get_el_sp(e) for e in element_list]
    ele1, ele2 = sorted(element_list, key=lambda x: x.atomic_radius)[:2]
    return ele1.atomic_radius + ele2.atomic_radius
