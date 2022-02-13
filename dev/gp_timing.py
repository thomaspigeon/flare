import numpy as np
import time
import os
from flare import gp, struc, predict
from flare.ase import calculator
from flare.ase.atoms import FLARE_Atoms
import multiprocessing as mp
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.build import bulk
from ase.build import make_supercell
from ase.calculators.eam import EAM
from numpy.random import rand


# Download an aluminum EAM potential from the NIST potential database.
eam_loc = "https://www.ctcms.nist.gov/potentials/Download/1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Al/2/Al99.eam.alloy"
wget_return = os.system("wget %s" % eam_loc)
if wget_return != 0:
    os.system("curl %s -o Al99.eam.alloy" % eam_loc)
eam_potential = EAM(potential="Al99.eam.alloy")

# Generate aluminum data.
a = 4.0
cell_ase = bulk('Al', 'fcc', a=a, cubic=True)
size = 2
P = [(size, 0, 0), (0, size, 0), (0, 0, size)]
super_cell = make_supercell(cell_ase, P)
super_cell.calc = eam_potential
pos = np.copy(super_cell.positions)

n_strucs = 10
n_atoms = len(super_cell)
species = [13] * n_atoms
jit_size = 0.1
positions = np.zeros((n_strucs, n_atoms, 3))
forces = np.zeros((n_strucs, n_atoms, 3))
cell = np.array(super_cell.cell)
print(cell)
for n in range(n_strucs):
    rand_jit = (rand(n_atoms, 3) * 2 - 1) * jit_size
    pos_curr = pos + rand_jit 
    super_cell.positions = pos_curr
    positions[n] = pos_curr
    forces[n] = super_cell.get_forces()

# Create a 2+3-body Gaussian process.
kernels = ['twobody', 'threebody']
component = 'mc'
hyps = np.array([0.1, 0.1, 0.1, 2.0, 0.5])  # initial (bad) choice of hyps
cutoffs = {'twobody': 7.0, 'threebody': 5.5}  # cutoff radii in A
maxiter = 100  # max number of hyperparameter optimziation steps

gp_model = gp.GaussianProcess(
  kernels=kernels,
  component=component,
  hyps=hyps,
  cutoffs=cutoffs,
  maxiter=50
)

# Create FLARE calculator.
flare_calc = calculator.FLARE_Calculator(gp_model, par=True)

# Put a few snapshots in the training set.
# snapshots = [500, 1500]
snapshots = [5]
for snapshot in snapshots:
    # create flare structure
    training_positions = positions[snapshot]
    training_forces = forces[snapshot]
    training_structure = struc.Structure(cell, species, training_positions)

    # add the structure to the training set of the GP
    gp_model.update_db(training_structure, training_forces)

gp_model.set_L_alpha()

# Create a validation structure.
validation_snapshot = 6
validation_positions = positions[validation_snapshot]
validation_forces = forces[validation_snapshot]
validation_structure = struc.Structure(cell, species, validation_positions)

# Predict forces serially.
time1 = time.time()
pred_forces, stds = \
    predict.predict_on_structure(validation_structure, gp_model)
time2 = time.time()
ser_f = time2 - time1

print(ser_f)

# Predict forces in parallel.
time1 = time.time()
pred_forces, stds = \
    predict.predict_on_structure_par(validation_structure, gp_model)
time2 = time.time()
par_f = time2 - time1

print(par_f)

# Predict energy, forces, and stresses serially.
time1 = time.time()
return_vals = \
    predict.predict_on_structure_efs(validation_structure, gp_model)
time2 = time.time()
ser_efs = time2 - time1

print(ser_efs)

# Predict energy, forces, and stresses in parallel.
time1 = time.time()
return_vals = \
    predict.predict_on_structure_efs_par(validation_structure, gp_model)
time2 = time.time()
par_efs = time2 - time1

print(par_efs)

# Predict with calculator.
atoms = validation_structure.to_ase_atoms()
flare_atoms = FLARE_Atoms.from_ase_atoms(atoms)
flare_atoms.calc = flare_calc
time1 = time.time()
flare_atoms.get_forces()
time2 = time.time()
calc_efs = time2 - time1

print(calc_efs)

# Take MD step with Velocity Verlet engine.
md_engine = VelocityVerlet(
    atoms=flare_atoms,
    timestep=1.0 * units.fs,
    )

time1 = time.time()
flare_calc.reset()
flare_atoms.calc = flare_calc
md_engine.step()
time2 = time.time()
md_step = time2 - time1

print(md_step)
