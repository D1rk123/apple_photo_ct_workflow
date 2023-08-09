import numpy as np
from pathlib import Path
from flexdata import display, data
from flextomo import projector
from flexcalc import process, analyze
import matplotlib

from folder_locations import get_data_folder

def reconstruct(input_folder, output_folder, spectrum, i, bh_correction = True, compound = 'H2O', density = 0.6):
    proj, geom = process.process_flex(input_folder, sample = 1, skip = 1, correct='cwi-flexray-2022-05-31')
    
    vol = projector.init_volume(proj)
    projector.FDK(proj, vol, geom)
    
    if bh_correction == True:
        density = density
        compound = compound
        energy, spec = analyze.calibrate_spectrum(proj, vol, geom, compound = compound, density = density, iterations=10000)
        print("=======================================")
        print(f"energy = {energy}")
        print(f"spec = {spec}")
        print("=======================================")
        spectrum[i-1, :] = spec
    
if __name__ == "__main__":
    #Stop plots from displaying
    #matplotlib.use('Agg')
    
    base_path = get_data_folder()
    proj_path = base_path / "projections"
    spectrum = np.zeros((120, 10), dtype=np.float64)

    #for subfolder in root_folder.iterdir():
    for i in range(1,121):
        reconstruct(proj_path / f"{i}", proj_path / f"{i}", bh_correction = True, spectrum = spectrum, i = i)
    
    mean_spec = np.mean(spectrum, axis=0)
    mean_spec /= np.sum(mean_spec)
    print(f"normalized mean spectrum = \n{mean_spec}")
    np.savetxt(base_path / "estimated_spectrums.csv", spectrum, delimiter=",")

