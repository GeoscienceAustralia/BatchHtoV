#!/usr/bin/env bash
#PBS -P vy72 
#PBS -q expressbw
#PBS -l walltime=5:00:00
#PBS -l mem=128GB
#PBS -l jobfs=1GB
#PBS -l ncpus=1
#PBS -l software=make 
#PBS -l wd 
#module load agdc-py2-prod/1.2.2
#module load pyqt
export KO='60'
export PATH=/g/data/ha3/lsd547/miniconda2/bin:$PATH
export PYTHONPATH=/g/data/ha3/lsd547/miniconda2/:/g/data/ha3/lsd547/mlpy/lib/python2.7/site-packages/:/g/data/ha3/lsd547/stockwell_transform/lib/python2.7/site-packages/
#source activate seismicpy27
INDIR=/g/data/ha3/Passive/OvernightData/Passive_Seismic_\(HVSR\)/Barrygowan/BarrygowanB2/BAGB2miniSEED/
#python process_raydec_median_derivative_cwt_oversampled.py $INDIR 100 0.1 28.0 200 $KO log BAR1_raydec_md_cwt_os_01_28 16
python process_median_bootstrap_derivative_oversampled.py 'raydeccwtlog3' $INDIR 100 0.1 50.0 200 $KO log BAR1_md_rd3_b2_os_01_50 16 > raydec3.log
