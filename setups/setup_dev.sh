#!/bin/bash


#setup lsst_sims
#source /global/common/software/lsst/cori-haswell-gcc/stack/setup_w_2018_13-sims_2_7_0.sh
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/sims_2_8_0//loadLSST.bash
setup lsst_sims

export PYTHONPATH=Sim_SNCosmo:$PYTHONPATH
export PYTHONPATH=Sim_SNSim:$PYTHONPATH
export PYTHONPATH=Sim_SNAna:$PYTHONPATH

export PYTHONPATH=../SN_Utils/Utils:$PYTHONPATH

#checking whether hdf5 is accessible localy or not
lib='h5py'
thedir='../lib/*/site-packages/'
echo $thedir
if [ -d ${thedir}$lib ]
then
    echo $lib 'already installed -> updating PYTHONPATH'
else
    echo $lib 'not installed -> installing with pip'
    pip install --prefix=../ ${lib}==2.7.1
    thedir='../lib/*/site-packages/'
fi
final_dir=`echo $thedir`
export PYTHONPATH=${final_dir}:$PYTHONPATH
