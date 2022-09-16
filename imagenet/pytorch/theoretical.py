"""
Theoretical maximum throughput that can be achieved for batch prediction.
This does not include reading or preprocessing.

We create a tensor of batch size x image dimensions
Use the largest batch size that can fit in GPU RAM (of p3.2xlarge instance)
Move the tensor to GPU.

Then feed the tensor in ~20 times to get the throughput.
"""
import time
import torch
from torchvision.models import resnet50, ResNet50_Weights

BATCH_SIZE = 1024 # Largest batch size for a V100 (~16384MiB)

start_time = time.time()
with torch.inference_mode():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    model.to("cuda")

    tensor = torch.rand(size=(BATCH_SIZE, 3, 224, 224), device="cuda")

    total_images = 0
    iterations = 20
    for _ in range(iterations):
        _ = model(tensor)
        total_images += BATCH_SIZE
end_time = time.time()

print(f"Total throughput: {total_images/(end_time-start_time)} (img/sec)")

# ~1000 images/sec

