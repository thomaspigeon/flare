import numpy as np
import time
import os
from flare import gp, struc, predict
from flare.ase import calculator
from flare.ase.atoms import FLARE_Atoms
import multiprocessing as mp

# Load AgI data.
AgI_location = "https://zenodo.org/record/3688843/files/AgI_data.zip?download=1"
wget_return = os.system("wget %s" % AgI_location)
# For Macs:
if wget_return != 0:
    os.system("curl %s -o AgI_data.zip?download=1" % AgI_location)
os.system("unzip -o AgI_data.zip?download=1")

# Load AIMD training data.
data_directory = 'AgI_data/'
species = np.load(data_directory + 'species.npy')  # atomic numbers of the atoms
positions = np.load(data_directory + 'positions.npy')  # in angstrom (A)
cell = np.load(data_directory + 'cell.npy')  # 3x3 array of cell vectors (in A)
forces = np.load(data_directory + 'forces.npy')  # in eV/A

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
snapshots = [500, 1500]
for snapshot in snapshots:
    # create flare structure
    training_positions = positions[snapshot]
    training_forces = forces[snapshot]
    training_structure = struc.Structure(cell, species, training_positions)

    # add the structure to the training set of the GP
    gp_model.update_db(training_structure, training_forces)

gp_model.set_L_alpha()

# Create a validation structure.
validation_snapshot = 2300
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
