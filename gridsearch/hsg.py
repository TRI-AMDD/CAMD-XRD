import itertools
import numpy as np
import json
from pymatgen.core import Lattice, Structure
from pymatgen.core.sites import PeriodicSite
from pyxtal.symmetry import get_wyckoffs
from tqdm.notebook import tqdm
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.analysis.diffraction import xrd

class HierarchicalStructureGeneration:
    """
    **Input:** Space group, lattice parameters, number of atoms for each element-type in the unit cell, and pair-wise minimum distances allowed.

    **Algorithm:**
    1. Get Wyckoff sites (symmtery operations) for the space group (attention should be paid to the spg setting used in experiments)
    2. Generate all possible combinations of Wyckoff sites satisfying the number of atoms (composition) in the cell. Filter out combinations that duplicate zero-degree-of-freedom Wyckoff sites.
    3. Sort combinations such that those with fewer number of distinct Wyckoff sites are higher on the list.
    4. For each combination on the list:
        1. Start with the Wyckoff positions of the element-type (e.g. "Pb") that has fewer number of atoms. For each Wyckoff position of this element in the combination:
            1. Generate sets of atomic positions from each Wyckoff position's symmetry operations for each point on the [x,y,z] grid with a specified step size within [0,1] (considering which dimensions are active)
            2. Remove sets of positions where positions overlap or closer than the distance threshold for the same (A-A) pairs.
        2. Repeat the overlap/distance check accross different Wyckoff sites of the same element. Skip those with atoms too close.
        3. If this is the first element-type considered (e.g. "Pb"), continue loop.
        4. If this is not the first element-type considered (e.g. "S"): Repeat the overlap/distance check between the newest element expanded in the loop (e.g. "S") and element-types already generated. (e.g. Pb). Skip those with atoms too close.
        5. Store structures that pass all distance filters.

    **Returns:** A list of feasible sets of atomic positions that satisfy the composition, symmetry and distance constraints.
    """

    def __init__(
        self, spg, a, b, c, alpha, beta, gamma, atoms, d_tol, d_mins=None, use_asu=False
    ):
        """
        Args:
            - spg (int): space group
            - a, b, c  (float): a, b, c  lattice parameters in Angstroms in conventional standard cell
            - alpha, beta, gamma (float): alpha, beta, gamma angles in conventional standard cell
            - atoms (list): List of tuples species in the form [# of atoms, 'Species'] e.g. [(4,'Pb'), (16,'O'), (4,'S')]
            - d_tol (float): general minimum distance between atoms in cell in Angstroms
            - d_mins (dict): Element pair specific minimum distances bewtween atoms in cell e.g. {'Pb': 1.5*2, 'S': 1.70*2, 'O': 2.1, 'O-Pb': 2.4, 'Pb-S': 3.0}
            - use_asu (bool): consider asymmetric unit
        Returns:
            HierarchicalStructureGeneration object
        
        """
        self.spg = spg
        self.a, self.b, self.c = a, b, c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.atoms = sorted(atoms)
        self.d_tol = d_tol
        self.d_mins = d_mins if d_mins else {}
        self.lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.use_asu = use_asu
        self.d_tol_squared = self.d_tol ** 2  # general overlap distance threshold
        self.d_mins_squared = {k: v ** 2 for k, v in self.d_mins.items()}
        self.wyckoffs = get_wyckoffs(spg)
        self.multiplicities = [len(w) for w in self.wyckoffs]
        self.final_strucs = None

        if use_asu:
            # TODO: Need to have package data management in the long term
            with open("asu_data.json", "r") as f:
                asu_data = json.load(f)
            self.ppipe = parse_asu(asu_data[str(self.spg)])
        else:
            self.ppipe = None

        self.all_active_wyckoffs = np.array(
            [
                i
                for i in range(len(self.wyckoffs))
                if sum(self.active_dim(self.wyckoffs[i])) > 0
            ]
        )
        filtered_strucs = []
        for i in itertools.product(
            *[self.get_possible_combinations(a[0]) for a in self.atoms]
        ):
            counter = np.zeros(len(self.wyckoffs))
            for j in i:
                for k in j:
                    counter[k[0]] += 1
            t = np.argwhere(counter > 1).flatten()
            if False not in np.isin(t, self.all_active_wyckoffs):
                filtered_strucs.append(i)

        self.filter_combinations = filtered_strucs
        self.filter_combinations = sorted(
            self.filter_combinations, key=lambda x: sum([len(i) for i in x])
        )

    def get_wyckoff_candidates(self, pos, d_min_squared, density):
        """
        This function generates all unique sets of atomic positions from a Wyckoff position's symmetry operations,
        on a grid of allowed free-varibles (with *n* equi-distant points on [0,1]) (e.g. n x n x n grid if all a 
        site has all x, y and z as free-variables; or nx1xn if x and z are free variables). It takes into account
        a minimum allowed distance.
        Args:
            pos (list): Symmetry operations of the Wyckoff position
            npoints (int): number of equi-distant points on [0,1] for generation of the free-variable grid.
            density (float): number of grid points per angstrom rounded up
            d_min_squared (float): If given, squared distance between each atom pair in the generated set of positions
                is compared to filter out unphysical structures.
        Returns:
            A set of *sets of atomic positions* generated satisfying the symmetry and distance constraints.

        """

        grid_xyz = []

        for i in range(3):
            if self.active_dim(pos)[i]:
                if self.use_asu:
                    npoints = int(
                        np.ceil(
                            (
                                self.lattice.parameters[i] * self.ppipe[0][i][1]
                                - self.lattice.parameters[i] * self.ppipe[0][i][0]
                            )
                            * density
                        )
                    )
                    print("Using {} points in direction {}".format(npoints, i))
                    grid_xyz.append(
                        get_linspace(
                            *self.ppipe[0][i],
                            npoints=npoints,
                            include_bounds=self.ppipe[1][i]
                        )
                    )
                else:
                    npoints = int(np.ceil((self.lattice.parameters[i]) * density))
                    grid_xyz.append(np.linspace(0, 1, npoints, endpoint=True))
            else:
                grid_xyz.append([0])  # if dimension is not active.

        candidates = []
        # forming a meshgrid for free x, y, and/or z parameters for the wyckoff site.
        for xyz in itertools.product(grid_xyz[0], grid_xyz[1], grid_xyz[2]):
            wyckoff_positions = []
            for so in pos:  # apply symmetry operations of the wyckoff
                product = so.operate(xyz)  # applies both rotation and translation.
                warped = self.warp(
                    product
                )  # make sure sites remain within the unit cells
                wyckoff_positions.append(tuple(warped))
            else:
                wyckoff_positions = frozenset(
                    wyckoff_positions
                )  # forming a set will get rid of duplciates (overlaps)

                if len(wyckoff_positions) == len(
                    pos
                ):  # if no overlapping sites, store set of wyckoff positions
                    skip_str = False
                    if d_min_squared:
                        for s1, s2 in itertools.combinations(wyckoff_positions, 2):
                            d2 = pbc_shortest_vectors(self.lattice, s1, s2, return_d2=True)[1]
                            if d2 < d_min_squared:
                                skip_str = True
                                break
                    if skip_str:
                        continue
                    candidates.append(wyckoff_positions)
        self.candidates = set(candidates)
        return self.candidates

    def get_possible_combinations(self, target_n_atoms):
        """
        Helper function to find all possible combinations of Wyckoff positions of a given space group
             that would satisfy the target atom count.
        Args:
            target_n_atoms (int): number of atoms to fit to Wyckoff sites
        Returns:
            list of tuples with Wyckoff site multiplicities that add up to target_n_atoms
        """
        self.combinations = [
            q
            for i in range(int(target_n_atoms/min(self.multiplicities)), 0, -1)
            for q in itertools.combinations_with_replacement(
                enumerate(self.multiplicities), i
            )
            if sum([k[1] for k in q]) == target_n_atoms
        ]
        return self.combinations
    def get_pymatgen_structure(self, struc):
        elems = []              
        for atom in self.atoms:
            for num in range(atom[0]):
                elems.append(atom[1])
        coords = []
        for coord in struc:
            coords = coords + coord
        return Structure(self.lattice,elems,coords)

    @staticmethod
    def active_dim(pos):
        """
        Helper function that checks if the wyckoff position has free variables
        Args:
            pos (list of Pymatgen SymmOp objects): Wyckoff symmetry operations of a spacegroup
        Returns:
            True if Wyckoff site contains degrees of freedom for atom position

        """
        return np.abs(pos[0].rotation_matrix).sum(axis=0) != 0

    @staticmethod
    def warp(coord):
        """
        Translates fractional coordinates to the cell closest closest to the origin.
        """
        for i in range(3):
            coord[i] = coord[i] % 1
        return coord

    def get_random_struc(self, combination_index):
        """
        Gets a random structure of a specified wyckoff configuration
        Args:
            combination_index (int): index of attribute filter_combinations which corresponds to the desired Wyckoff sites

        Returns:
            Random Pymategen structure with given Wyckoff sites
        """

        coords = []
        for combin in self.filter_combinations[combination_index]:
            xyz = np.random.rand(3)
            for atom in combin:
                wyckoff_ops = self.wyckoffs[atom[0]]
                for op in wyckoff_ops:
                    coords.append(self.warp(op.operate(xyz)))

        species = []
        for element in self.atoms:
            for multiplicity in range(element[0]):
                species.append(element[1])

        return Structure(self.lattice, species, coords)

    def get_structure_grid(self, density, combination=0):
        """
        Combining groups of wyckoff: sites satisfying composition requirements, and filtering
        out those that would repeat wyckoffs that do not have internal degree of freedom (hence can't be occupied
        by two different species.
        Args:
            density (float): number of grid points per angstrom rounded up

        Returns:
            list of structure coordinates which span the grid
        """

        wyckoffs = []
        combin = self.filter_combinations[combination]
        for elem in range(len(combin)):
            for atom in combin[elem]:
                wyckoffs.append(atom[0])
        if len(wyckoffs) > 1:
            combs = []
            for comb in itertools.combinations(wyckoffs,2):
                combs.append(comb)

            combs = frozenset(combs)

            wyckoff_overlaps = {}
            wyckoff_lengths = {}

            for atom in tqdm(combs): 
                wyckoff_overlaps[tuple(sorted(atom))] = {}


                _d_tol_squared = self.d_tol_squared

                wyckoff_grid1 = self.get_wyckoff_candidates(
                                    pos=self.wyckoffs[sorted(atom)[0]],
                                    d_min_squared=self.d_tol_squared,
                                    density=density)

                wyckoff_grid2 = self.get_wyckoff_candidates(
                                    pos=self.wyckoffs[sorted(atom)[1]],
                                    d_min_squared=self.d_tol_squared,
                                    density=density)
                
                if sorted(atom)[0] not in wyckoff_lengths:
                    wyckoff_lengths[sorted(atom)[0]] = len(wyckoff_grid1)
                if sorted(atom)[1] not in wyckoff_lengths:
                    wyckoff_lengths[sorted(atom)[1]] = len(wyckoff_grid2)
                
                count1 = -1
                for struct1 in wyckoff_grid1:
                    count1+=1
                    count2 = -1
                    for struct2 in wyckoff_grid2:
                        count2+=1
                        if self.wyckoffs[atom[0]] == self.wyckoffs[atom[1]] and count2 <= count1:
                            continue
                        skip_str = False
                        for s1, s2 in itertools.product(*[struct1,struct2]):
                            d2 = pbc_shortest_vectors(self.lattice, s1, s2, return_d2=True)[1]
                            if d2 < self.d_tol_squared:
                                skip_str = True
                                break
                            
                        if not skip_str:
                            wyckoff_overlaps[tuple(sorted(atom))][(count1, count2)] = True

        else:
            wyckoff_grid1 = self.get_wyckoff_candidates(
                                    pos=self.wyckoffs[wyckoffs[0]],
                                    d_min_squared=self.d_tol_squared,
                                    density=density)
            wyckoff_lengths = {wyckoffs[0]: len(wyckoff_grid1)}        
        if len(wyckoffs) == 1:
            good_wyckoff_combinations = []
            for i in range(wyckoff_lengths[wyckoffs[0]]):
                good_wyckoff_combinations.append((i,))

        else:
            first_pair =  tuple(sorted(wyckoffs[0:2]))
            if wyckoffs[1] > wyckoffs[0]:
                good_wyckoff_combinations = list(wyckoff_overlaps[first_pair].keys())
            else:
                good_wyckoff_combinations = []
                for combination in list(wyckoff_overlaps[first_pair].keys()):
                    good_wyckoff_combinations.append(tuple(reversed(combination)))
            if len(wyckoffs) > 2:
                for atom in tqdm(range(2,len(wyckoffs))):
                    possible_grids = wyckoff_lengths[wyckoffs[atom]]
                    new_good_wyckoff_combinations = []
                    prev_atoms = wyckoffs[0:atom]
                    
                    for struct in good_wyckoff_combinations:
        #                 print(struct)
                        
                        for grid in range(possible_grids):
                            good_str = True
                            for index in range(len(frozenset(prev_atoms))):
                                pair = tuple(sorted((prev_atoms[index],wyckoffs[atom])))
                                grid_pair = (struct[index], grid)

                                if grid_pair not in wyckoff_overlaps[tuple(sorted(pair))] and \
                                tuple(reversed(grid_pair)) not in wyckoff_overlaps[tuple(sorted(pair))]:
                                    good_str = False
                                    break
                                if not good_str:
                                    break
                            else:
                                new_good_wyckoff_combinations.append(struct+(grid,))
                    
                    good_wyckoff_combinations = new_good_wyckoff_combinations

        wyckoff_grids = []
        for i in range(len(wyckoffs)):
            wyckoff_grids.append(list(self.get_wyckoff_candidates(
                                pos=self.wyckoffs[wyckoffs[i]],
                                d_min_squared=self.d_tol_squared,
                                density=density)))
            
        final_strucs = []
        for combs in good_wyckoff_combinations:
            count = 0
            final_strucs.append([])
            for grid in range(len(combs)):
                final_strucs[-1].append(list(wyckoff_grids[count][combs[grid]]))
                count += 1 

        return final_strucs

    def get_structure_grids(
        self,
        density,
        combinations=None,
        parallel=False,
        n_jobs=-1,
        batch_size=100000,
        backend="loky",
    ):
        """
        Generates structures corresponding to a list of Wyckoff site combinations satisfying the compositional
        requirements. This method simply calls the proper get_structure_grid method for all combinations listed
        as argument. This is the preferred convenience method of structure generation in general.
        Args:
            density (float): number of grid points per angstrom rounded up
            combinations (list): Indices of Wyckoff combinations (available from the filter_combinations attribute
                of the class). Defaults to None, which generated the grids for *all* combinations listed in
                filter_combinations
            parallel (bool): Switches the generation method to a parallelized version. See method
                parallel_get_structure_grid to decide if parallelization is feasible for the particualr use case.
            n_jobs (int): number of processes or threads to use. defaults to -1 (all).
            batch_size (int, str): see joblib.Parallel
            backend (str): see joblib.Parallel

        Returns:
            A group of lists of structure grids for each combination listed in combinations
        """

        final_strucs = []
        combinations = (
            combinations if combinations else range(len(self.filter_combinations))
        )
        for combination in combinations:
            if not parallel:
                final_strucs.append(self.get_structure_grid(density, combination))
            else:
                final_strucs.append(
                    self.parallel_get_structure_grid(
                        density, combination, n_jobs, batch_size, backend
                    )
                )
        self.final_strucs = final_strucs
        return final_strucs


def get_linspace(ll, ul, npoints=10, include_bounds=None):
    """
    Helper function to construct linspace obeying inequality conditions at limits
    """
    if not include_bounds:
        include_bounds = [True, True]
    endpoint = False
    if include_bounds[1]:
        endpoint = True
    _np = npoints
    _ = 0
    if not include_bounds[0]:
        _np = npoints + 1
        _ = 1
    return np.linspace(ll, ul, _np, endpoint=endpoint)[_:]


def parse_asu(p):
    """
    Helper function to parse limits from asu_data
    """

    def _t(t):
        if t[0] == "<":
            val = eval(t.split("<")[1].replace("=", ""))
        else:
            val = eval(t.split("<")[0].replace("=", ""))
        return val, "<=" in t

    c = ["x", "y", "z"]
    lims, ends = [], []
    for i in range(3):
        l, e = [], []
        for t in p[i].split(c[i]):
            m, r = _t(t)
            l.append(m)
            e.append(r)
        lims.append(l)
        ends.append(e)
    return lims, ends

def get_Rs(two_thetas, intensities, matpatterns):
        xrd_calc = xrd.XRDCalculator()
        rietweld_mat = []

        referencex = two_thetas
        referencey = intensities
        for matindex in matpatterns:
            numerator = 0
            count = 0

            for twotheta in range(len(referencex)):
                peak_intensity = 0

                for twotheta2 in range(len(matindex.x)):
                    if count == len(referencex)-1:
                        if np.abs(matindex.x[twotheta2] - referencex[twotheta]) <= 0.15 and \
                        np.abs(matindex.x[twotheta2] - referencex[twotheta-1]) > np.abs(matindex.x[twotheta2] - referencex[twotheta]): 
                            peak_intensity += matindex.y[twotheta2]
                    elif count == 0:
                        if np.abs(matindex.x[twotheta2] - referencex[twotheta]) <= 0.15 and \
                        np.abs(matindex.x[twotheta2] - referencex[twotheta+1]) > np.abs(matindex.x[twotheta2] - referencex[twotheta]): 
                            peak_intensity += matindex.y[twotheta2]
                    else:
                        if np.abs(matindex.x[twotheta2] - referencex[twotheta]) <= 0.15 and \
                        np.abs(matindex.x[twotheta2] - referencex[twotheta+1]) > np.abs(matindex.x[twotheta2] - referencex[twotheta]) and\
                        np.abs(matindex.x[twotheta2] - referencex[twotheta-1]) > np.abs(matindex.x[twotheta2] - referencex[twotheta]): 
                            peak_intensity += matindex.y[twotheta2]

                numerator += (peak_intensity - referencey[twotheta])**2

                count += 1
            total = np.sum(matindex.y)

            rietweld_mat.append(numerator/total)
        return(rietweld_mat)





