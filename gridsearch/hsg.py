import itertools
import numpy as np
import json
from pymatgen import Lattice
from pyxtal.symmetry import get_wyckoffs
from tqdm.notebook import tqdm
from pymatgen.core import Structure
from joblib import Parallel, delayed


# TODO: asu was not functioning properly so I made it optional (and also changed, please fix if needed.


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
        # if self.use_asu:
        #         continue

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
                            if (
                                np.sum(
                                    (
                                        self.lattice.get_cartesian_coords(np.array(s1))
                                        - self.lattice.get_cartesian_coords(
                                            np.array(s2)
                                        )
                                    )
                                    ** 2
                                )
                                < d_min_squared
                            ):
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
            for i in range(len(self.multiplicities), 0, -1)
            for q in itertools.combinations_with_replacement(
                enumerate(self.multiplicities), i
            )
            if sum([k[1] for k in q]) == target_n_atoms
        ]
        return self.combinations

    @staticmethod
    def active_dim(pos):
        """
        Helper function that checks if the wyckoff position has free variables
        Args:
            pos (list of Pymatgen SymmOp objects): Wyckoff symmetry operations of a spacegroup
        Returns:
            True if Wyckoff site contains degrees of freedom for atom position

        """
        return pos[0].rotation_matrix.sum(axis=0) != 0

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

        final_strucs = []

        rolling_good_base_strs = []

        combin = self.filter_combinations[combination]
        print(combination, combin)
        for atom in range(len(combin)):
            elem_group = combin[atom]
            elem = self.atoms[atom][1]

            if elem in self.d_mins_squared:
                d_min_squared = self.d_mins_squared[elem]
            else:
                d_min_squared = None

            _d_tol_squared = d_min_squared if d_min_squared else self.d_tol_squared
            # FIRST WE WILL GET WYCKOFF SITE GRIDS;
            # AND REMOVE THOSE OVERLAP ACCROSS DIFFERENT SITES FOR SAME ATOM!
            _g = []
            print("{}.{}: Elem self loop: {}".format(combination, combin, elem))
            passed = []
            for site1 in elem_group:

                # check for exchange duplicates if there is the same wyckoff site within the same element
                multiple = 0
                if site1 in passed:
                    continue
                for site2 in elem_group:

                    if site1 == site2:
                        multiple += 1
                passed.append(site1)

                if multiple == 1:
                    _g.append(
                        list(
                            self.get_wyckoff_candidates(
                                pos=self.wyckoffs[site1[0]],
                                d_min_squared=d_min_squared,
                                density=density,
                            )
                        )
                    )
                else:
                    new_list = []

                    for wyckoff_groups in itertools.combinations(
                        self.get_wyckoff_candidates(
                            pos=self.wyckoffs[site1[0]],
                            d_min_squared=d_min_squared,
                            density=density,
                        ),
                        multiple,
                    ):
                        good_str = 1
                        for group in itertools.product(*wyckoff_groups):

                            for s1, s2 in itertools.combinations(group, 2):
                                if (
                                    np.sum(
                                        (
                                            self.lattice.get_cartesian_coords(
                                                np.array(s1)
                                            )
                                            - self.lattice.get_cartesian_coords(
                                                np.array(s2)
                                            )
                                        )
                                        ** 2
                                    )
                                    < _d_tol_squared
                                ):
                                    good_str = 0
                                    break
                            if good_str == 0:
                                break
                        else:
                            new_list.append(frozenset().union(*wyckoff_groups))

                    _g.append(new_list)

            within_elem_group = list(itertools.product(*_g))

            good_strs_within_elem_group = []
            for struct in tqdm(within_elem_group):
                skip_str = False
                for sub_pairs in itertools.combinations(struct, 2):
                    for s1, s2 in itertools.product(*sub_pairs):
                        if (
                            np.sum(
                                (
                                    self.lattice.get_cartesian_coords(np.array(s1))
                                    - self.lattice.get_cartesian_coords(np.array(s2))
                                )
                                ** 2
                            )
                            < _d_tol_squared
                        ):
                            skip_str = True
                            break
                    else:
                        continue
                    break
                if not skip_str:
                    good_strs_within_elem_group.append(
                        [[i for sub in struct for i in sub]]
                    )
            if atom == 0:
                rolling_good_base_strs = good_strs_within_elem_group

            # NOW; we will combine good_strs_within_elem_group and rolling_good_base_strs and
            # remove if any bad structures accross these.

            if atom > 0:
                print("{}.{}:  Elem pairs loop: {}".format(combination, combin, elem))
                good_structures_merged = []
                for structs in tqdm(
                    itertools.product(
                        rolling_good_base_strs, good_strs_within_elem_group
                    ),
                    total=len(rolling_good_base_strs)
                    * len(good_strs_within_elem_group),
                ):
                    skip_str = False
                    for i in range(len(structs[0])):
                        # different atoms of previous kind
                        atomgroup1 = structs[0][i]
                        atomgroup2 = structs[1][0]
                        pair = "-".join(sorted([self.atoms[i][1], self.atoms[atom][1]]))
                        _d_tol_squared = max(
                            self.d_mins_squared.get(pair, 0), self.d_tol_squared
                        )
                        for s1, s2 in itertools.product(atomgroup1, atomgroup2):
                            if (
                                np.sum(
                                    (
                                        self.lattice.get_cartesian_coords(np.array(s1))
                                        - self.lattice.get_cartesian_coords(
                                            np.array(s2)
                                        )
                                    )
                                    ** 2
                                )
                                < _d_tol_squared
                            ):
                                skip_str = True
                                break
                        if skip_str:
                            break
                    if not skip_str:
                        good_structures_merged.append(structs[0] + [structs[1][0]])
                rolling_good_base_strs = good_structures_merged
        final_strucs += rolling_good_base_strs

        self.final_strucs = final_strucs
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

    def parallel_get_structure_grid(
        self, density, combination=0, n_jobs=-1, batch_size=100000, backend="loky",
    ):
        """
        Warning: this is the parallel version of get_structure_grids. Since the atomic tasks are extremely fast
        get_structure_grids can be much faster if npoints is small. If point density is large, by dispatching large
        batches (e.g. 100k) - notable parallelization speedup might happen as overhead is overcome.

        Args:
            density (float): number of grid points per angstrom rounded up (point density)
            combination (int): Index of the Wyckoff combination (available from the filter_combinations attribute
                of the class).
            n_jobs (int): number of processes or threads to use. defaults to -1 (all).
            batch_size (int, str): see joblib.Parallel
            backend (str): see joblib.Parallel

        Returns:
            list of structure coordinates which span the grid
        """

        final_strucs = []
        combin = self.filter_combinations[combination]

        print(combination, combin)
        rolling_good_base_strs = []
        for atom in range(len(combin)):
            elem_group = combin[atom]
            elem = self.atoms[atom][1]
            if elem in self.d_mins_squared:
                d_min_squared = self.d_mins_squared[elem]
            else:
                d_min_squared = None

            _d_tol_squared = d_min_squared if d_min_squared else self.d_tol_squared
            # FIRST WE WILL GET WYCKOFF SITE GRIDS;
            # AND REMOVE THOSE OVERLAP ACCROSS DIFFERENT SITES FOR SAME ATOM!
            _g = []
            print("{}.{}: Elem self loop: {}".format(combination, combin, elem))
            passed = []
            for site1 in elem_group:
                multiple = 0
                if site1 in passed:
                    continue
                for site2 in elem_group:

                    if site1 == site2:
                        multiple += 1
                passed.append(site1)

                if multiple == 1:
                    _g.append(
                        list(
                            self.get_wyckoff_candidates(
                                pos=self.wyckoffs[site1[0]],
                                d_min_squared=d_min_squared,
                                density=density,
                            )
                        )
                    )
                else:
                    new_list = []

                    for wyckoff_groups in itertools.combinations(
                        self.get_wyckoff_candidates(
                            pos=self.wyckoffs[site1[0]],
                            d_min_squared=d_min_squared,
                            density=density,
                        ),
                        multiple,
                    ):
                        good_str = 1
                        for group in itertools.product(*wyckoff_groups):

                            for s1, s2 in itertools.combinations(group, 2):
                                if (
                                    np.sum(
                                        (
                                            self.lattice.get_cartesian_coords(
                                                np.array(s1)
                                            )
                                            - self.lattice.get_cartesian_coords(
                                                np.array(s2)
                                            )
                                        )
                                        ** 2
                                    )
                                    < _d_tol_squared
                                ):
                                    good_str = 0
                                    break
                            if good_str == 0:
                                break
                        else:
                            new_list.append(frozenset().union(*wyckoff_groups))

                    _g.append(new_list)

            within_elem_group = list(itertools.product(*_g))

            good_strs_within_elem_group = Parallel(
                n_jobs=n_jobs, batch_size=batch_size, backend=backend, verbose=1
            )(
                delayed(struct_func0)(struct, _d_tol_squared, self.lattice)
                for struct in within_elem_group
            )

            good_strs_within_elem_group = [_ for _ in good_strs_within_elem_group if _]

            if atom == 0:
                rolling_good_base_strs = good_strs_within_elem_group

            # NOW; we will combine good_strs_within_elem_group and rolling_good_base_strs and
            # remove if any bad structures accross these.

            if atom > 0:
                print("{}.{}:  Elem pairs loop: {}".format(combination, combin, elem))
                rolling_good_base_strs = Parallel(
                    n_jobs=n_jobs, batch_size=batch_size, backend=backend, verbose=1
                )(
                    delayed(struct_func)(
                        structs,
                        atom,
                        self.atoms,
                        self.d_mins_squared,
                        self.d_tol_squared,
                        self.lattice,
                    )
                    for structs in itertools.product(
                        rolling_good_base_strs, good_strs_within_elem_group
                    )
                )
                rolling_good_base_strs = [_ for _ in rolling_good_base_strs if _]
        final_strucs += rolling_good_base_strs

        self.final_strucs = final_strucs
        return final_strucs

    def get_structure_vecs(self):
        pass


def struct_func0(struct, _d_tol_squared, lattice):
    skip_str = False
    for sub_pairs in itertools.combinations(struct, 2):
        for s1, s2 in itertools.product(*sub_pairs):
            if (
                np.sum(
                    (
                        lattice.get_cartesian_coords(np.array(s1))
                        - lattice.get_cartesian_coords(np.array(s2))
                    )
                    ** 2
                )
                < _d_tol_squared
            ):
                skip_str = True
                break
        else:
            continue
        break
    if not skip_str:
        return [[i for sub in struct for i in sub]]


def struct_func(structs, atom, atoms, d_mins_squared, d_tol_squared, lattice):
    skip_str = False
    for i in range(len(structs[0])):
        # different atoms of previous kind
        # print(atom, atoms)
        atomgroup1 = structs[0][i]
        atomgroup2 = structs[1][0]
        pair = "-".join(sorted([atoms[i][1], atoms[atom][1]]))
        _d_tol_squared = max(d_mins_squared.get(pair, 0), d_tol_squared)
        for s1, s2 in itertools.product(atomgroup1, atomgroup2):
            if (
                np.sum(
                    (
                        lattice.get_cartesian_coords(np.array(s1))
                        - lattice.get_cartesian_coords(np.array(s2))
                    )
                    ** 2
                )
                < _d_tol_squared
            ):
                skip_str = True
                break
        if skip_str:
            break
    if not skip_str:
        return structs[0] + [structs[1][0]]
    else:
        return None


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
