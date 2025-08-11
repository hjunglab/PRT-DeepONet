PRT-DeepONet Models
This repository contains PRT-DeepONet models for three different reactive transport problems in porous media:

PRT-DeepONet_Irreversible_Sorption – Predicts steady-state solute concentration fields for irreversible reactions based on the geometry of the porous medium and dimensionless parameters.

PRT-DeepONet_Reversible_Sorption – Predicts the time evolution of solute concentration fields for reversible reactions using the geometry of the porous medium and dimensionless parameters.

PRT-DeepONet_Monod – Predicts the time evolution of concentration fields for nonlinear reactions following Monod kinetics.

All three versions share the same execution structure for consistency and reproducibility, differing only in the model architecture details and dataset dimensions.
The networks follow the DeepONet framework, consisting of:

BranchCNN: Encodes the porous geometry.

BranchFNN: Encodes physical parameters (e.g., Pe, Da).

Trunk: Encodes spatial-temporal coordinates and GDF features.
