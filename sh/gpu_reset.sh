# ~/bin/gpu_refresh.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Stop known GPU users (optional)"
sudo systemctl stop ollama 2>/dev/null || true
# If headless over SSH and you don't need local GUI, you *may* stop gdm:
# sudo systemctl stop gdm 2>/dev/null || sudo systemctl stop gdm3 2>/dev/null || true

echo "[2/5] Ensure no processes hold /dev/nvidia*"
if fuser -v /dev/nvidia* 2>/dev/null; then
  echo ">> Some processes still hold the GPU. Kill them or stop their services, then re-run."
  exit 1
fi

echo "[3/5] Unload + reload NVIDIA modules (vGPU-safe)"
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

echo "[4/5] Toggle persistence"
sudo nvidia-smi -pm 0 || true
sudo nvidia-smi -pm 1 || true

echo "[5/5] Quick sanity"
nvidia-smi --query-gpu=pstate,clocks.sm,utilization.gpu,power.draw --format=csv -i 0
echo "Done."
