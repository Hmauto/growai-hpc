#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT_NAME>     # <-- REPLACE with your account (e.g., tra24_...)
#SBATCH -p boost_usr_prod          # GPU Partition
#SBATCH --time=00:30:00            # Run for 30 mins
#SBATCH -N 1                       # 1 Node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # Request 1 NVIDIA A100 GPU
#SBATCH --mem=64G                  # 64GB RAM
#SBATCH --job-name=ollama_test
#SBATCH --output=ollama_output_%j.log

# --- 1. SETUP ENVIRONMENT ---
# Point to your custom bin folder
export PATH=$WORK/bin:$PATH
# Point to Scratch for storage (avoid Home quota issues)
export OLLAMA_MODELS=$SCRATCH/ollama_models
# Allow server to listen (required on some setups)
export OLLAMA_HOST=0.0.0.0:11434

echo "--------------------------------------"
echo "Running on host: $(hostname)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "--------------------------------------"

# --- 2. START OLLAMA SERVER ---
echo "Starting Ollama Server..."
ollama serve &
SERVER_PID=$!

# Wait for it to wake up
sleep 15

# --- 3. RUN AI ---
echo "Pulling Llama3 model (this takes time only the first run)..."
ollama pull llama3

echo "Sending prompt to Llama3..."
ollama run llama3 "Explain what High Performance Computing is in 2 sentences."

# --- 4. CLEANUP ---
# Kill the server when done so the job finishes cleanly
kill $SERVER_PID
