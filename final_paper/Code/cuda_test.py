import torch

def test_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test GPU computation
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        print("\nPerforming test computation on GPU...")
        z = torch.matmul(x, y)
        print("GPU computation successful!")
    else:
        print("CUDA is not available. Check PyTorch installation.")

if __name__ == "__main__":
    test_gpu()