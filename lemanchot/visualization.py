
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from enum import Enum
from typing import List, Optional, Union
from itertools import cycle
import torch

class COLORS(Enum):
    """
    Enumeration of basic colors for plotting and visualizing multilabel
    predictions (class maps).
    """

    BaseColor = [0, 0, 0]
    SweetBrown = [170, 57, 57]  # background
    DarkGold = [170, 108, 57]  # crack
    MyrtleGreen = [34, 102, 102]  # non-crack
    ForestGreen = [45, 136, 45]  # screws-rivets
    Melon = [255, 170, 170]
    Feldspar = [255, 209, 170]
    DesaturatedCyan = [102, 153, 153]
    DarkSeaGreen = [136, 204, 136]
    FuzzyWuzzy = [212, 106, 106]
    Tan = [212, 154, 106]
    Ming = [64, 127, 127]
    Apple = [85, 170, 85]
    FaluRed = [128, 21, 21]
    Russet = [128, 69, 21]
    DeepJungleGreen = [13, 77, 77]
    RoyalGreen = [17, 102, 17]
    DarkChocolate = [85, 0, 0]
    SealBrown = [85, 39, 0]
    SacramentoStateGreen = [0, 51, 51]
    DarkGreen = [0, 68, 0]

    @classmethod
    def values(cls) -> List[List[int]]:
        """
        Generate a list containing the RGBA values of the available colors.

        Returns:
            List[List[int]]: list of RGBA values of the available colors.
        """
        return list(map(lambda c: c.value, cls))

    @classmethod
    def names(cls) -> List[str]:
        """
        Generate a list containing the names of the available colors.

        Returns:
            List[str]: list of names of the available colors.
        """
        return list(map(lambda c: c.name, cls))


def mask2colormap(
    input: torch.Tensor, underlayer: Optional[Union[None, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Generates a color map tensor with 4 channels (RGBA). This function uses the
    colors available in the COLORS Enum without randomization.

    Args:
        input (torch.Tensor): Input ONE-HOT encoded tensor representing class assignements.
                              Expects (N, C, H, W).
        underlayer ([None, torch.Tensor]): Optional parameter. If not None, this tensor will be
                                         used as an underlayer for the color map. Meaning the input pixels
                                         without any classes will be replaced by the underlayer pixels.
                                         Defaults to None. Expects (N, 3 or 4, H, W) and must match input dims.

    Raises:
        RuntimeError: Wrong input and/or underlayer dimensions.

    Returns:
        torch.Tensor: Color map or combined color map of the one-hot encoded class assignement.
    """

    if input.dim() != 3:
        raise RuntimeError(f"Expects input to have 3 dimensions, got {input.dim()}.")

    ch, h, w = input.size()
    
    aval_colors = torch.tensor(
        COLORS.values(), device=input.device, dtype=torch.int32
    ).view(-1, 3, 1, 1)
    base_color = aval_colors[0]
    
    output = torch.zeros(size=(3, h, w), device=input.device, dtype=torch.int32)
    input = input.bool()

    for i, c in zip(range(ch), cycle(aval_colors[1:, ...])):
        mask = input[i, ...].repeat(3, 1, 1)
        output += torch.where(mask, c, base_color)

    output = torch.clamp(output, 0, 255)

    if underlayer is not None:
        if underlayer.shape[-2:] != output.shape[-2:]:
            raise RuntimeError(
                "Expected input and underlay to have the same height and width,"
                f" got {output.shape[-2:]} and {underlayer.shape[-2:]}."
            )
        if underlayer.device != input.device:
            underlayer.to(input.device)

        return torch.clamp((underlayer + output), 0, 255).byte()

    return output.byte()