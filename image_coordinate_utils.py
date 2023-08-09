import torch

def volume_lookup(volume, coords, padding_mode="border"):
    lookup_coords = ((coords+0.5)/torch.flipud(torch.tensor(volume.size(), device=coords.device))[None, :]) * 2 - 1

    lookup = torch.nn.functional.grid_sample(
        volume[None, None, ...],
        lookup_coords[None, :, None, None, :],
        align_corners=False,
        padding_mode=padding_mode
    )
    return lookup.squeeze()
    
def slice_to_volume_coords(coords):
    coords = torch.cat([torch.zeros((len(coords), 1), dtype=torch.float32), coords], dim=1)
    coords = torch.flip(coords, dims=(1, ))
    return coords
