#!/bin/bash


#setup lsst_sims
source /global/common/software/lsst/cori-haswell-gcc/stack/w.2018.19_sim2.8.0/loadLSST.bash
setup lsst_sims

export PYTHONPATH=Sim_SNCosmo:$PYTHONPATH
export PYTHONPATH=Sim_SNSim:$PYTHONPATH
export PYTHONPATH=Sim_SNAna:$PYTHONPATH

export PYTHONPATH=../SN_Utils/Utils:$PYTHONPATH
