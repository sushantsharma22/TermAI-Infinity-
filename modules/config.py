def check_environment():
    """
    Check if GPU is available and print a message. 
    We do not require a GPU, but it's helpful to inform the user.
    """
    import torch
    print("[TermAI Infinity] Environment check...")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for inference.")
    else:
        print("No GPU detected. Running on CPU (may be slower).")
