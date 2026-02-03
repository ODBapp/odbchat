import os
import platform
import subprocess


def _has_cuda() -> bool:
    try:
        subprocess.check_call("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def install_llama_cpp() -> None:
    system = platform.system()
    has_cuda = _has_cuda()
    print(f"Detected OS: {system}")
    print("CUDA detected." if has_cuda else "CUDA not detected.")

    base_cmd = ["uv", "pip", "install", "--upgrade", "--no-cache-dir", "llama-cpp-python>=0.3.2"]
    env = os.environ.copy()

    if system == "Darwin":
        # Apple Silicon: prefer Metal build
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        env["FORCE_CMAKE"] = "1"
        print("Installing llama-cpp-python with Metal support...")
        subprocess.check_call(base_cmd, env=env)
        return

    if has_cuda and system == "Linux":
        # Try CUDA wheels first, then fall back to source build
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        wheel_cmd = base_cmd + ["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"]
        try:
            print("Attempting CUDA wheel install...")
            subprocess.check_call(wheel_cmd, env=env)
            return
        except subprocess.CalledProcessError:
            print("CUDA wheel failed, falling back to source build...")
            subprocess.check_call(base_cmd, env=env)
            return

    print("Installing CPU version of llama-cpp-python...")
    subprocess.check_call(base_cmd, env=env)


def main() -> None:
    print("Setting up odbchat environment...")
    install_llama_cpp()
    print("Environment setup complete.")


if __name__ == "__main__":
    main()
