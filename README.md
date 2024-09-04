# Recall-tempo-of-Hebbian-sequences
Code for reproducing the plots of the manuscript "Recall tempo of Hebbian sequences depends on the interplay of Hebbian kernel with tutor signal timing"

To run the simulations, use run.py. This script takes a run number as an argument for the purpose of parallelization. For instance,

python run.py --run-num 2

Will run a simulation corresponding with the second set of parameters for each plot. To run all of the simulations in parallel with slurm, do something like

sbatch [PARAMETERS] --array=0-50 python run.py --run-num ${SLURM_ARRAY_TASK_ID}

Once this is finished, to generate the plots run

python run.py

This will run all of the simulations in serial, but since they have already been run the previous result will be loaded. Simulation outputs are stored in overlaps_cache and make_plots_cache; delete these if a fresh run is desired. It is also possible to run everything in serial by skipping the parallelization step and simply calling "python run.py", but this will take a while (about 24 hrs, excluding the supplementary figures).

Supplementary Figures S3-S6 require ~80GB of memory to run and require a lot of time (a few days or so). Comment out the appropriate lines in run.py to exclude these simulations

