import torch.nn as nn
from chgnet.model import CHGNet

class CHGNet_feature_gen(nn.Module):

    def __init__(self, use_device = None):
        import CHGNet
        chgnet = CHGNet.load()
        if use_device is not None:
            chgnet.to(use_device)
        self.chgnet = chgnet

    def get_crystal_fea(
        self,
        structure: Structure | list[Structure],
        batch_size: int = 128,
        verbose=False,
    ) -> Tensor:
        """CGCNN predict formatio nenergy from pymatgen structure.

        Args:
            structure(Structure): structure to be predicted
            batch_size(int): batch size for CGCNN prediction
        Returns: formation energy (eV/atom) of provided structure

        """
        self.model.eval()