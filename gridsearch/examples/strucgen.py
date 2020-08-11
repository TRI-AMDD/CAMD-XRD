import camd 
import hsg

from tqdm.notebook import tqdm
from pymatgen.analysis.diffraction import xrd
from pymatgen import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen import Lattice
from monty.serialization import loadfn, dumpfn
mpr = MPRester('xpFvqo6ae6RNF3rqM0WI')

from ase import Atom, Atoms
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
import lammps
from ase import io
from pymatgen import MPRester
from monty.serialization import dumpfn, loadfn 
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.lammpslib import LAMMPSlib
import os
from ase import io
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators import kim
from monty.os import cd

from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import utils
import time
print('hello', hsg.__file__)

start = time.time()
mpid = 'mp-7631'
os.mkdir('{}'.format(mpid))
density = 1.5
original_str = SpacegroupAnalyzer(mpr.get_structure_by_material_id(mpid)).get_conventional_standard_structure()
# SETUP
spg = SpacegroupAnalyzer(original_str).get_symmetry_dataset()['number']
a=original_str.lattice.a ; b=original_str.lattice.b; c=original_str.lattice.c 
alpha=original_str.lattice.alpha; beta=original_str.lattice.beta; gamma=original_str.lattice.gamma
print(spg,a,b,c,alpha,beta,gamma)
tracker = {}
for specie in original_str.species:
    if specie.as_dict()['element'] not in tracker:
        tracker[specie.as_dict()['element']] = 1
    else:
        tracker[specie.as_dict()['element']] += 1

atoms = [(tracker[key], key) for key in tracker]

# General ovelap threshold:
d_tol = 1.5
# Specific thresholds:
# d_mins = {'Ag': 1.5*2, 'Ta': 1.70*2, 'O': 2.1, 'O-Ag': 2.4, 'Ta-O': 2.85}
d_mins = utils.get_minimum_distances('Si6C6', tol=0.0, glue = 'Si')


# d_mins = {}
# Numer of steps in [0,1] for each free (x, y or z) parameter.

generator = hsg.HierarchicalStructureGeneration(spg,a,b,c,alpha,beta,gamma,atoms,d_tol,d_mins, True)
print(generator.filter_combinations)
struc_coords =generator.get_structure_grids(density,range(len(generator.filter_combinations)))
    
all_strucs = []
lattice = Lattice.from_parameters(a,b,c,alpha,beta,gamma)
# species = ['Si']*8+['O']*16
# species = ['Pb']*4+['S']*4+['O']*16
# species = ['Au']*16+ ['O']*24
# species = ['Al']*2+ ['N']*6+['Ti']*8

sorteddict = list(reversed(sorted(tracker)))

species = []
for specie in sorteddict:
    for num in range(tracker[specie]):
        species.append(specie) 
dumpfn(struc_coords, '{}/struc_coords-{}-density={}.json'.format(mpid, mpid,density))

for l in range(len(struc_coords)):
    all_strucs.append([])
    for m in range(len(struc_coords[l])):
        
        sites = []
        for j in list(struc_coords[l][m]):
            for k in j:
                sites.append(k)
        all_strucs[-1].append(Structure(lattice, species, sites))
print(species,atoms)

print('Structure Generation: ', time.time()-start)

def get_Rs(standard_struc, matpatterns):
    xrd_calc = xrd.XRDCalculator()
    rietweld_mat = []
    
    reference = xrd_calc.get_pattern(standard_struc)
    for matindex in matpatterns:
        numerator = 0
        count = 0

    #     if np.all(matindex.x != xrd_py[mat]['xrd'].x):
    #         continue
        for twotheta in range(len(reference.x)):
            peak_intensity = 0

            for twotheta2 in range(len(matindex.x)):
                if count == len(reference.x)-1:
                    if np.abs(matindex.x[twotheta2] - reference.x[twotheta]) <= 0.15 and \
                    np.abs(matindex.x[twotheta2] - reference.x[twotheta-1]) > np.abs(matindex.x[twotheta2] - reference.x[twotheta]): 
                        peak_intensity += matindex.y[twotheta2]
                elif count == 0:
                    if np.abs(matindex.x[twotheta2] - reference.x[twotheta]) <= 0.15 and \
                    np.abs(matindex.x[twotheta2] - reference.x[twotheta+1]) > np.abs(matindex.x[twotheta2] - reference.x[twotheta]): 
                        peak_intensity += matindex.y[twotheta2]
                else:
                    if np.abs(matindex.x[twotheta2] - reference.x[twotheta]) <= 0.15 and \
                    np.abs(matindex.x[twotheta2] - reference.x[twotheta+1]) > np.abs(matindex.x[twotheta2] - reference.x[twotheta]) and\
                    np.abs(matindex.x[twotheta2] - reference.x[twotheta-1]) > np.abs(matindex.x[twotheta2] - reference.x[twotheta]): 
                        peak_intensity += matindex.y[twotheta2]
    #                     if twotheta == 7:
    #                         print(matindex.x[twotheta2], xrd_py[mat]['xrd'].x[twotheta])
    #                         print(np.abs(matindex.x[twotheta2] - xrd_py[mat]['xrd'].x[twotheta+1]), np.abs(matindex.x[twotheta2] - xrd_py[mat]['xrd'].x[twotheta]))
    #                         print(matindex.y[twotheta2], xrd_py[mat]['xrd'].y[twotheta], matindex.x[twotheta2], xrd_py[mat]['xrd'].x[twotheta])

            numerator += (peak_intensity - reference.y[twotheta])**2

    #         if count3 >= 5:
    #             print(mat)
    #             print(count3)
            count += 1
        total = np.sum(matindex.y)

        rietweld_mat.append(numerator/total)
    return(rietweld_mat)

matpatterns_all = []
for struc in all_strucs:
#     matpatterns_all.append([])
    xrd_calc = xrd.XRDCalculator()
    patterns =  Parallel(n_jobs=-1)(delayed(xrd_calc.get_pattern)(struc[i]) for i in range(len(struc)))
    matpatterns_all.append(patterns)
print('XRD Patterns: ', time.time()-start)

Rvalues = Parallel(n_jobs=-1)(delayed(get_Rs)(original_str, matpatterns_all[i]) for i in range(len(matpatterns_all)))

print('Rvalues: ', time.time()-start)

dumpfn(Rvalues, '{}/rvalues-{}-density={}.json'.format(mpid,mpid,density))


from pymatgen.analysis.structure_matcher import StructureMatcher
sm = StructureMatcher(ltol=0.6,stol=0.6, angle_tol=25)

scores = []
for strucs in all_strucs:
    scoresstruc =  Parallel(n_jobs=-1)(delayed(sm.get_rms_dist)(strucs[m], original_str) for m in range(len(strucs)))
    scores.append(scoresstruc)
    
siscores = []
for i in range(len(scores)):
    siscores.append([])
    for j in range(len(scores[i])):
        if scores[i][j] == None:
            siscores[-1].append(None)
        else:
            siscores[-1].append(scores[i][j][0])
print('RMS Scores: ', time.time()-start)

dumpfn(siscores, '{}/scores-{}-density={}.json'.format(mpid,mpid, density))


from ase import Atom, Atoms
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
import lammps
from ase import io
from pymatgen import MPRester
from monty.serialization import dumpfn, loadfn 
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.lammpslib import LAMMPSlib

import os
from ase import io
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.kim import kim

# os.environ['ASE_LAMMPSRUN_COMMAND'] = 'lmp_serial'
mpr = MPRester('xpFvqo6ae6RNF3rqM0WI')

# parameters = {'pair_style': 'tersoff',
#               'pair_coeff': ['* * 2007_SiO.tersoff O Si']}
# files = ['2007_SiO.tersoff']
# calc = LAMMPS(parameters=parameters, files=files, specorder=['O', 'Si'])


# calc = LAMMPS(parameters=parameters, files=files, specorder=['O', 'Si'])
# calc  = kim.KIM('Sim_LAMMPS_MEAM_ZhangTrinkle_2016_TiO__SM_513612626462_000')
# calc  = kim.KIM('Sim_LAMMPS_EDIP_JiangMorganSzlufarska_2012_SiC__SM_435704953434_000')
# calc  = kim.KIM('Sim_LAMMPS_MEAM_AlmyrasSangiovanniSarakinos_2019_NAlTi__SM_871795249052_000')
# calc  = kim.KIM('Sim_LAMMPS_IFF_PCFF_HeinzMishraLinEmami_2015Ver1v5_FccmetalsMineralsSolventsPolymers__SM_039297821658_000')
# calc = kim.KIM("LJ_ElliottAkerson_2015_Universal__MO_959249795837_003")
# calc = kim.KIM('Sim_LAMMPS_BOP_MurdickZhouWadley_2006_GaAs__SM_104202807866_000')
# calc = kim.KIM('Tersoff_LAMMPS_ErhartJuslinGoy_2006_ZnO__MO_616776018688_002')
# calc = kim.KIM('Sim_LAMMPS_ModifiedTersoff_ByggmastarHodilleFerro_2018_BeO__SM_305223021383_000')
# calc = kim.KIM('EAM_Dynamo_NicholAckland_2016_Na__MO_048172193005_000')
# calc = kim.KIM('Sim_LAMMPS_BOP_MurdickZhouWadley_2006_GaAs__SM_104202807866_000')
calc = kim.KIM('Sim_LAMMPS_EDIP_JiangMorganSzlufarska_2012_SiC__SM_435704953434_000')
orig_atoms=AseAtomsAdaptor.get_atoms(original_str)

siatoms = [] 
for i in range(len(all_strucs)):
    siatoms.append([])
    for j in range(len(all_strucs[i])):
        siatoms[-1].append(AseAtomsAdaptor.get_atoms(all_strucs[i][j]))
        
sienergies = []
for i in range(len(siatoms)):
    sienergies.append([])
    for j in range(len(siatoms[i])):
        siatoms[i][j].set_calculator(calc)
        sienergies[-1].append(siatoms[i][j].get_potential_energy())

print('Energies: ', time.time()-start)


# dumpfn(all_strucs, 'all_strucs.json')
dumpfn(sienergies, '{}/energies-{}-density={}.json'.format(mpid,mpid, density))

# print(time.time()-start)