#!/usr/bin/env bash
#PBS -P vy72 
#PBS -q expressbw
#PBS -l walltime=0:30:00
#PBS -l mem=128GB
#PBS -l jobfs=1GB
#PBS -l ncpus=1
#PBS -l software=make 
#PBS -l wd 
#module load agdc-py2-prod/1.2.2
#module load pyqt
export PATH=/g/data/ha3/lsd547/miniconda2/bin:$PATH
export PYTHONPATH=/g/data/ha3/lsd547/miniconda2/:/g/data/ha3/lsd547/mlpy/lib/python2.7/site-packages/:/g/data/ha3/lsd547/stockwell_transform/lib/python2.7/site-packages/
source activate seismicpy27
INDIR=/g/data/ha3/Passive/OvernightData/Passive_Seismic_\(HVSR\)/Barrygowan/BarrygowanB2/BAGB2miniSEED/
python process_hvsr_interp_error.py 'cwtlog' $INDIR 200 0.1 50.0 1000 50 log BAR1_cwt_001_50_32omega 32
