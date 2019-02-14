# SN_Catalog_Simulations

To run the simulation:

- get the code:
   - git clone https://github.com/lsstdesc/SN_Catalog_Simulations.git
   - git clone https://github.com/lsstdesc/SN_Utils.git

- use the dev branch (for now)
  - cd SN_Catalog_Simulations ; git checkout dev
  - cd SN_Utils ; git checkout dev_stable

- cd ../SN_Catalog_Simulations
- source setups/setup.sh
- python sn_simulation/sn_simulation.py input/param.yaml

N.B: This software needs lsst_sim. It is currently running @NERSC with w.2018.19_sim2.8.0.