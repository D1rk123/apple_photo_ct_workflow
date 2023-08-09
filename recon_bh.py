import numpy as np
from pathlib import Path
from flexdata import display, data
from flextomo import projector
from flexcalc import process, analyze
import matplotlib

from folder_locations import get_data_folder

def reconstruct(input_folder, save_path, energy, spec, bh_correction = True, compound = 'H2O', density = 0.6):
    '''Reconstructs the volume using flexbox. Slices will be written to recon/ subfolder.
    '''

    # Don't redo reconstructions
    #if save_path.exists():
    #    return
        
    proj, geom = process.process_flex(input_folder, sample = 1, skip = 1, correct='cwi-flexray-2022-05-31')

    save_path.mkdir(exist_ok=True)
    
    vol = projector.init_volume(proj)
    projector.FDK(proj, vol, geom)
    
    if bh_correction == True:
        proj_cor = process.equivalent_density(proj, geom, energy, spec, compound = compound, density = density, preview=False)
        vol_rec = np.zeros_like(vol)
        projector.FDK(proj_cor, vol_rec, geom)
        vol = vol_rec
    
    data.write_stack(save_path, 'slice', vol, dim = 0)
    
if __name__ == "__main__":
    #Stop plots from displaying
    matplotlib.use('Agg')
    
    base_path = get_data_folder()
    proj_path = base_path / "projections"
    save_path = base_path / "fbp_bh_corrected_recons"
    
    energy = np.linspace(5, 100, 10)
    spec = np.array([2.50715155e-04, 3.80118462e-01, 1.51004538e-01, 1.73835060e-01,
    1.32302499e-01, 8.74416199e-02, 5.10978500e-02, 2.29350857e-02,
    1.01416913e-03, 0.00000000e+00])

    for i in range(1,121):
        reconstruct(proj_path / f"{i}", save_path / f"{i}", energy, spec, bh_correction = True)

