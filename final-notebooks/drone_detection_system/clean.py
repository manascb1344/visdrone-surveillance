import gc, torch

# Run garbage collection
gc.collect()

# Clear cache on all GPUs
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
