import numpy as np
import pandas as pd
import itertools
from pymatgen import Element, Composition


def get_minimum_distances(formula, tol=0.15, glue=None, polya=None):
    """
    Generates a dictionary of allowed minimum distances for ionic/covalent structures. It allows
    incorporation of some basic chemical insights.
    Examples:
        (i)   'PbCuBiS3', glue='S'
        (ii)  'LiFePO4', glue='O', polya=['O','P']
        (iii) 'Li3PS4', glue='S', polya=['P','S']
        (iv)  'PbSO4', glue='O', polya=['S','O']
        (v)   'La4S3N2', glue=['S','N']
    Args:
        - formula (str): chemical formula of the compound.
        - tol (float): tolerance factor over the distance limits calculated from ionic radii.
        - glue (str) or (list): element(s) that can be viewed as filling in-between the cations.
            Often the most abundant element is the best to use. Defaults to None.
        - polya (list): elements that may form poly-anions. if given, further contracts their
            min separations one more time by tol.
    Returns:
        Dictionary of minimum distances allowed, compatible with HierarchicalStructureGeneration
    """
    if type(glue) is str:
        glue = [glue]
    polya = polya if polya else []

    c = Composition(formula)
    df = pd.DataFrame(c.oxi_state_guesses())
    r_df = df.copy()
    for i in df.columns:
        el = Element(i)
        rs = []
        for ox in df[i]:
            dat = np.array(list(el.data["Ionic radii"].keys()))
            match = dat[np.argmin(np.abs(dat.astype(float) - ox))]
            r = el.data["Ionic radii"][match]
            rs.append(r)
        r_df[i] = rs
    pair_dict = {}
    for i, j in itertools.combinations_with_replacement(list(c.as_dict().keys()), 2):
        if i == j:
            label = i
        else:
            label = "-".join(sorted([i, j]))

        if glue:
            if (i in glue) ^ (j in glue) or ((j in glue) and (i in glue)):
                pair_dict[label] = np.mean(r_df[i] + r_df[j]) * (1 - tol)
            else:
                pair_dict[label] = np.mean(
                    [
                        np.mean(
                            np.sqrt((r_df[i] + r_df[g]) ** 2 - 4 * (r_df[g] ** 2) / 3)
                            + np.sqrt((r_df[j] + r_df[g]) ** 2 - 4 * (r_df[g] ** 2) / 3)
                        )
                        * (1 - tol)
                        for g in glue
                    ]
                )
        else:
            pair_dict[label] = np.mean(r_df[i] + r_df[j]) * (1 - tol)

        if ((i in polya) and (j in polya)) and i != j:
            pair_dict[label] *= 1 - tol
    return pair_dict
