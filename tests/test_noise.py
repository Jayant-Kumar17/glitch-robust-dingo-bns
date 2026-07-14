from gwpy.timeseries import TimeSeries
import torch

print("Fetching 10 seconds of real LIGO O3 noise from GWOSC...")
# GPS time 1240559616 is a known clean segment from the O3 run
o3_noise = TimeSeries.fetch_open_data(
    "H1", 1240559616, 1240559626, verbose=True
)

# Convert it to a PyTorch tensor
noise_tensor = torch.tensor(o3_noise.value, dtype=torch.float32)

print(f"\nSuccess! Downloaded noise shape: {noise_tensor.shape}")
print(f"PyTorch using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
