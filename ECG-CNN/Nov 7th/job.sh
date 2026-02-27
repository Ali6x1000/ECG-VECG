

#==================================================================
# SLURM SCRIPT FOR TENSORFLOW/KERAS PYTHON JOB
#==================================================================

# --- Job Configuration ---
#SBATCH --job-name=ecg_tf_train        # A descriptive name for your job
#SBATCH --output=python_job_output.log # ⭐️ THIS saves all output to "python_job_output.log"
#SBATCH --error=python_job_error.log   # Saves errors to a separate file (good practice)

# --- Cluster-Specific Configuration (FROM YOUR EXAMPLE) ---
#SBATCH --account=csds438              # ⭐️ ADDED: Your class/project account
#SBATCH --partition=markov_gpu         # ⭐️ ADDED: The GPU partition name
#SBATCH --constraint=gpu2080          # ⭐️ ADDED: Requesting a specific node type
#SBATCH --gres=gpu:1                   # ⭐️ CRITICAL: Request 1 GPU

# --- Resource Requests ---
#SBATCH --time=02:00:00                # Request 2 hours (HH:MM:SS). ADJUST THIS!
#SBATCH --ntasks=1                     # Run one single task
#SBATCH --cpus-per-task=4              # Request 4 CPU cores (for data loading)
#SBATCH --mem=32G                      # Request 32GB of RAM. ADJUST THIS!

# --- Job Execution ---
echo "--- JOB STARTED ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
date


echo "Activating Python venv from /home/aan90/ali/ECG-VCG/my_tf_env"
source /home/aan90/ali/ECG-VCG/my_tf_env/bin/activate

echo "Environment loaded."
echo "Running Python script 'slurm.py'..."

# Run your Python script.
python -u slurm.py

echo "--- JOB FINISHED ---"
date

