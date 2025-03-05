#!/bin/bash
#SBATCH --job-name=luca_dughera_job      # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --time=24:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END, FAIL               # Send email at job completion
#SBATCH --mail-user=lucadughera@gmail.com    # Email address for notifications

#Load necessary modules (if needed)
#module load module_name

#Your job commands go here
#For example:
python test.py

#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt