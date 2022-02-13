import numpy as np
import time
import os
import gp, struc, kernels
from env import AtomicEnvironment
import multiprocessing as mp
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.build import bulk
from ase.build import make_supercell
from ase.calculators.eam import EAM
from numpy.random import rand

# Run with Feb. 24 commit:
# https://github.com/mir-group/flare/tree/9f6014694209c469f4509182fcc2e69dce25d039
# This script should be copied to the "otf_engine" directory.

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
kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.1, 0.1, 0.1, 2.0, 0.5])  # initial (bad) choice of hyps
cutoffs = np.array([7.0, 5.5])
maxiter = 100  # max number of hyperparameter optimziation steps

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs)

# Define serial prediction function (adapted from otf.py).
def predict_on_structure(structure, gp):
    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.absolute(var))
    
    return structure.forces

# Put a few snapshots in the training set.
# snapshots = [500, 1500]
snapshots = [5]
for snapshot in snapshots:
    # create flare structure
    training_positions = positions[snapshot]
    training_forces = forces[snapshot]
    training_structure = struc.Structure(cell, species, training_positions)

    # add the structure to the training set of the GP
    gp_model.update_db(training_structure, training_forces, custom_range=[0])

gp_model.set_L_alpha()

# Create a validation structure.
validation_snapshot = 6
validation_positions = positions[validation_snapshot]
validation_forces = forces[validation_snapshot]
validation_structure = struc.Structure(cell, species, validation_positions)

# Predict forces serially.
for n in range(10):
    time1 = time.time()
    forces = predict_on_structure(validation_structure, gp_model)
    time2 = time.time()
    ser_f = time2 - time1

    print(ser_f)
