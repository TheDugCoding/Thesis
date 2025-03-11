#!/bin/bash
#SBATCH --job-name=dug_job      # Job name
#SBATCH --output=pyg-multinode.log        # outputfile
#SBATCH --error=error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --time=24:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL               # Send email at job completion
#SBATCH --mail-user=lucadughera@gmail.com    # Email address for notifications

#Load necessary modules (if needed)
#module load module_name

#Your job commands go here
#For example:
srun python src/Scripts/Modeling/Pre_Training/Topological_Pre_Training/modeling_graphsage_unsup.py
export PYTHONPATH=$PYTHONPATH:/science-nfs-sys/vsm01/users/1738380/Desktop/Thesis/src


#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt