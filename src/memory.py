import torch


def model_size_in_bytes(model):
    total_bytes = 0
    for param in model.parameters():
        total_bytes += (
            param.nelement() * param.element_size()
        )  # num_elements * size_per_element
    for buffer in model.buffers():
        total_bytes += (
            buffer.nelement() * buffer.element_size()
        )  # num_elements * size_per_element
    return total_bytes


def print_model_memory_usage(model):
    size_bytes = model_size_in_bytes(model)
    size_mb = size_bytes / (1024**2)  # Convert from bytes to megabytes
    print(f"The model uses approximately {size_mb:.2f} MB")


def get_gpu_memory():
    if torch.cuda.is_available():
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs Available: {num_gpus}")

        # Print memory information for each GPU
        for i in range(num_gpus):
            torch.cuda.synchronize(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_memory = torch.cuda.mem_get_info(i)[0]
            used_memory = total_memory - free_memory

            print(f"GPU {i}:")
            print(f"  Total Memory: {total_memory / 1e9:.2f} GB")
            print(f"  Used Memory: {used_memory / 1e9:.2f} GB")
            print(f"  Free Memory: {free_memory / 1e9:.2f} GB")
    else:
        print("No CUDA-capable device is detected")
