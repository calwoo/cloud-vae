import torch


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"You have {torch.cuda.device_count()} GPUs")

    x = torch.rand(3, 4)
    print("This is a tensor: \n", x)
