#!/usr/bin/env bash
#PBS -P vy72 
#PBS -q express 
#PBS -l walltime=3:00:00
#PBS -l mem=8GB
#PBS -l jobfs=1GB
#PBS -l ncpus=1
#PBS -l software=make 
#PBS -l wd 
#module load agdc-py2-prod/1.2.2
#module load pyqt
export PATH=/g/data/ha3/lsd547/miniconda2/bin:$PATH
export PYTHONPATH=/g/data/ha3/lsd547/miniconda2/:/g/data/ha3/lsd547/mlpy/lib/python2.7/site-packages/:/g/data/ha3/lsd547/stockwell_transform/lib/python2.7/site-packages/
source activate seismicpy27
python runbatch.py cwt2 '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/AdventureWay1/aw01/AW01miniSEED/' \
    '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/AdventureWay1/aw01/AW01miniSEED/' \
    --nfreq 100 --fmin 0.1 --fmax 10.0 --resample-log-freq --output-prefix cwt_lb
