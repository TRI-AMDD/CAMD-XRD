import itertools
import numpy as np
from pymatgen import Structure
from pymatgen import Lattice
from pyxtal.symmetry import get_wyckoffs
from tqdm.notebook import tqdm

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
    def __init__(self, spg, a, b, c, alpha, beta, gamma, atoms, d_tol, d_mins=None, npoints=10):
        """
        Blah blah blah
        Args:
            - spg (int): space group
            - a (float): ....
            
        
        """
        self.spg = spg
        self.a,  self.b, self.c = a, b, c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.atoms = sorted(atoms)
        self.d_tol = d_tol
        self.d_mins = d_mins if d_mins else {}
        self.npoints = npoints
        
        self.lattice = Lattice.from_parameters(a,b,c,alpha,beta,gamma)
        
        self.d_tol_squared = self.d_tol**2 # general overlap distance threshold
        self.d_mins_squared = {k: v**2 for k,v in self.d_mins.items()}
        
        self.wyckoffs = get_wyckoffs(spg)
        self.multiplicities = [len(w) for w in self.wyckoffs]
        
        self.all_active_wyckoffs = np.array([i for i in range(len(self.wyckoffs)) if sum(self.active_dim(self.wyckoffs[i]))>0])
    
    def get_wyckoff_candidates(self, pos, d_min_squared):
        """
        This function generates all unique sets of atomic positions from a Wyckoff position's symmetry operations,
        on a grid of allowed free-varibles (with *n* equi-distant points on [0,1]) (e.g. n x n x n grid if all a 
        site has all x, y and z as free-variables; or nx1xn if x and z are free variables). It takes into account
        a minimum allowed distance.
        Args:
            pos (list): Symmetry operations of the Wyckoff position
            npoints (int): number of equi-distant points on [0,1] for generation of the free-variable grid.
            d_min_squared (float): If given, squared distance between each atom pair in the generated set of positions 
                is compared to filter out unphysical structures.
        Returns:
            A set of *sets of atomic positions* generated satisfying the symmetry and distance constraints.

        """

        grid_xyz=[]
        for i in self.active_dim(pos):
            if i:
                grid_xyz.append(np.linspace(0,1.0,self.npoints,endpoint=False))
            else:
                grid_xyz.append([0]) # if dimension is not active.

        candidates = []
        # forming a meshgrid for free x, y, and/or z parameters for the wyckoff site.
        for xyz in itertools.product(grid_xyz[0],
                                     grid_xyz[1],
                                     grid_xyz[2]):
            wyckoff_positions = []
            for so in pos: #apply symmetry operations of the wyckoff 
                product = so.operate(xyz) # applies both rotation and translation.
                warped = self.warp(product)    # make sure sites remain within the unit cells
                wyckoff_positions.append(tuple(warped))

            wyckoff_positions = frozenset(wyckoff_positions) # forming a set will get rid of duplciates (overlaps)

            if len(wyckoff_positions) == len(pos): # if no overlapping sites, store set of wyckoff positions
                skip_str = False
                if d_min_squared:
                    for s1,s2 in itertools.combinations(wyckoff_positions,2):
                        if np.sum((
                                    self.lattice.get_cartesian_coords(np.array(s1))
                                    -self.lattice.get_cartesian_coords(np.array(s2)))**2 )< d_min_squared:
                            skip_str = True
                            break
                if skip_str:
                    continue
                candidates.append(wyckoff_positions)
        self.candidates = set(candidates) 
        return self.candidates
    
    def get_possible_combinations(self, target_n_atoms):
        """
        Helper function to find all possible combinatios of Wyckoff positions of a given space group
             that would satisfy the target atom count.
        """
        self.combinations = [q for i in range(len(self.multiplicities), 0, -1) 
              for q in itertools.combinations_with_replacement(enumerate(self.multiplicities), i) if sum([k[1] for k in q]) == target_n_atoms]
        return self.combinations
    
    @staticmethod
    def active_dim(pos):
        """Helper function that checks if the wyckoff position has free variables"""
        return pos[0].rotation_matrix.sum(axis=0) != 0  
    
    @staticmethod
    def warp(coord):
        """
        Puts fractional coordinates that fall outside back into [0,1].
        """
        for i in range(3):
            coord[i]=coord[i]%1
        return coord
    
    
    def get_structure_grid(self, top_X_combinations = -1):
        """
        Combining groups of wyckoff: sites satisfying composition requirements, and filtering
        out those that would repeat wyckoffs that do not have internal degree of freedom (hence can't be occupied
        by two different species.
        """
        filter_combinations=[]
        for i in itertools.product(*[self.get_possible_combinations(a[0]) for a in self.atoms]):
            counter = np.zeros(len(self.wyckoffs))
            for j in i:
                for k in j:
                    counter[k[0]] +=1
            t = np.argwhere( counter> 1 ).flatten()
            if False not in np.isin(t,self.all_active_wyckoffs):
                filter_combinations.append(i)
        
        self.filter_combinations = filter_combinations
        self.filter_combinations = sorted(self.filter_combinations,key=lambda x: sum([len(i) for i in x]))
         
        filter_further = []

        counter=0
        for combin in self.filter_combinations[:top_X_combinations]:
            print(counter, combin)
            rolling_good_base_strs = []
            for atom in range(len(combin)):
                elem_group = combin[atom]
                elem = self.atoms[atom][1]
                if elem in self.d_mins_squared:
                    d_min_squared = self.d_mins_squared[elem]
                else:
                    d_min_squared = None

                # FIRST WE WILL GET WYCKOFF SITE GRIDS; 
                # AND REMOVE THOSE OVERLAP ACCROSS DIFFERENT SITES FOR SAME ATOM!
                _g = []
                print('{}.{}: Elem self loop: {}'.format(counter, combin, elem))
                for site in elem_group:
                    _g.append(list(self.get_wyckoff_candidates(pos=self.wyckoffs[site[0]],
                                                               d_min_squared=d_min_squared)))
                within_elem_group = list(itertools.product(*_g))

                _d_tol_squared = d_min_squared if d_min_squared else self.d_tol_squared
                
                good_strs_within_elem_group = []
                for struct in tqdm(within_elem_group):
                    skip_str = False
                    for sub_pairs in itertools.combinations(struct,2):
                        for s1,s2 in itertools.product(*sub_pairs):
                            if np.sum( (
                                            self.lattice.get_cartesian_coords(
                                                        np.array(s1))
                                                    -self.lattice.get_cartesian_coords(
                                                        np.array(s2)))**2 ) < _d_tol_squared:
                                skip_str = True
                                break
                        else:
                            continue
                        break
                    if not skip_str:
                        good_strs_within_elem_group.append([[i for sub in struct for i in sub]])

                if atom==0:
                    rolling_good_base_strs = good_strs_within_elem_group

                # NOW; we will combine good_strs_within_elem_group and rolling_good_base_strs and
                # remove if any bad structures accross these.

                if atom>0:
                    print('{}.{}:  Elem pairs loop: {}'.format(counter, combin, elem))
                    good_structures_merged = []
                    for structs in tqdm( itertools.product(rolling_good_base_strs, good_strs_within_elem_group), 
                                       total=len(rolling_good_base_strs)*len(good_strs_within_elem_group)):
                        skip_str = False
                        for i in range(len(structs[0])):
                            # different atoms of previous kind
                            atomgroup1 = structs[0][i]
                            atomgroup2 = structs[1][0]
                            pair = '-'.join(sorted([self.atoms[i][1], self.atoms[atom][1]]))
                            _d_tol_squared = max( self.d_mins_squared.get(pair, 0), self.d_tol_squared)     
                            for s1,s2 in itertools.product(atomgroup1,atomgroup2):
                                if np.sum( (self.lattice.get_cartesian_coords(
                                                            np.array(s1))
                                                        -self.lattice.get_cartesian_coords(
                                                            np.array(s2)))**2 ) < _d_tol_squared:
                                        skip_str = True
                                        break
                            if skip_str:
                                break
                        if not skip_str:
                            good_structures_merged.append(structs[0]+[structs[1][0]])
                    rolling_good_base_strs = good_structures_merged
            filter_further+=rolling_good_base_strs
            counter+=1
        
        self.filter_further = filter_further
        return filter_further
    
    def get_structure_vecs(self):
        pass
