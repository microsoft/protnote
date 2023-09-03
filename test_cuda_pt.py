import torch

def test_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check your installation.")

if __name__ == "__main__":
    test_cuda()
