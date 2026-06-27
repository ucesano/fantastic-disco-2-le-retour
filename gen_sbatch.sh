#!/bin/bash
# Generates one sbatch script per (matrix, ranks) combination and submits them.
# 10 matrices x {4,3,2} ranks = 30 jobs, each a single run that fits the 5-min cap.
set -euo pipefail

matrices=(
  cage15
  ASIC_680ks
  Ga41As41H72
  rajat31
  Rucci1
  Si41Ge41H72
  webbase-1M
  boyd2
  memchip
  Maragal_8
)

ranks=(4 3 2)

# --- Build once, as its own short job, so 30 jobs don't run `make` concurrently ---
# (Remove this block + the --dependency flag below if ./bin/spmv is already built.)
build_script="build.sh"
cat > "$build_script" << 'BUILD_EOF'
#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=spmv_build
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err

module load OpenMPI
module load CUDA/11.8.0
make
BUILD_EOF

build_id=$(sbatch --parsable "$build_script" | cut -d';' -f1)
echo "Build job: $build_id"

# --- One job per (matrix, ranks); each waits for the build to succeed ---
for mtx in "${matrices[@]}"; do
  for np in "${ranks[@]}"; do
    script="sbatch_${mtx}_${np}.sh"
    cat > "$script" << JOB_EOF
#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=spmv_${mtx}_${np}
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=${np}
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:${np}
#SBATCH --output=logs/${mtx}_${np}.out
#SBATCH --error=logs/${mtx}_${np}.err

module load OpenMPI
module load CUDA/11.8.0

mpirun -np ${np} ./bin/spmv mtx/${mtx}.mtx
JOB_EOF
  sbatch --dependency=afterok:"$build_id" "$script"
  done
done
