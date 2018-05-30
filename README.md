# SN_Catalog_Simulations

To run the simulation:

- get the code:
   - git clone https://github.com/lsstdesc/SN_Catalog_Simulations.git
   - git clone https://github.com/lsstdesc/SN_Utils.git

- use the dev branch (for now)
  - cd SN_Catalog_Simulations ; git checkout dev
  - cd SN_Utils ; git checkout dev

- cd ../SN_Catalog_Simulations
- source setups/setup.sh
- python SN_Simulation/sn_simulation.py input/param.yaml