#!/bin/bash
#SBATCH --job-name=GAT_TRAINING_Synthetic      # Job name
#SBATCH --output=pyg-multinode.log        # outputfile
#SBATCH --error=error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --time=72:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL               # Send email at job completion
#SBATCH --mail-user=lucadughera@gmail.com    # Email address for notifications

#Load necessary modules (if needed)
#module load module_name

#Your job commands go here
#For example:

export PROJECT_PATH=/science-nfs-sys/vsm01/users/1738380/Desktop/Thesis
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH

srun python $PROJECT_PATH/src/modeling/pre_training/topological_pre_training/deep_graph_infomax.py


#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt