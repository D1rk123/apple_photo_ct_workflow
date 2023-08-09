import numpy as np
import scipy
import re
from matplotlib import pyplot as plt

from folder_locations import get_data_folder, get_results_folder
from click_core_endpoints import read_slice_csv

def get_unusable_slices(scan_number, photo_metadata):
    raw_str_split = photo_metadata[scan_number].split(",")
    if raw_str_split[1] == "0":
        return None
    if raw_str_split[2] == "":
        return []
    return [int(s) for s in raw_str_split[2].split(" ")]

def read_scale(file_path):
    with open(file_path, "r") as params_file:
        lines = params_file.readlines()
    return float(lines[5])
    
def get_usable_slices(photo_path, scan_name, unusable_slices):
    photo_nrs = []    
    photo_nrs_raw = sorted([int(re.findall(r'\d+', path.name)[1]) for path in photo_path.glob(f"Kanzi{scan_name}_slice_*.png")])

    for photo_nr in photo_nrs_raw:
        file_name = f"Kanzi{scan_name}_slice_{photo_nr}.png"
        if photo_nr not in unusable_slices:
            photo_nrs.append(photo_nr)

    photo_nrs = np.array(photo_nrs)
    
    return photo_nrs

if __name__ == "__main__":
    voxel_size = 0.129302

    scans_folder = get_data_folder()
    csv_path = scans_folder / "selected_point_compare_slice.csv"
    
    photo_metadata_path = scans_folder / "photo_metadata.csv"

    with open(photo_metadata_path) as file:
        photo_metadata = [line.rstrip() for line in file]
    
    selected_slices = read_slice_csv(csv_path)

    results_folder = get_results_folder()
    photo_annotations_folder = results_folder / "2023-03-05_click_core_endpoints_4" / "annotated_points"
    ct_annotations_folder = results_folder / "2023-03-07_click_ct_core_endpoints_1" / "annotated_points"
    registration_parameters_folder = results_folder / "2023-03-05_calc_kanzi_ct_slices_1" / "registration_parameters"
    
    per_apple_distances = {}
    all_distances = []
    
    for apple_nr, photo_nr in selected_slices:
        #print(f"Slice {(apple_nr, photo_nr)}")
        try:
            photo_points = np.genfromtxt(photo_annotations_folder / f"apple_{apple_nr}_photo_{photo_nr}.csv", delimiter=",", ndmin=2)
            ct_points = np.genfromtxt(ct_annotations_folder / f"apple_{apple_nr}_photo_{photo_nr}.csv", delimiter=",", ndmin=2)
            scale = read_scale(registration_parameters_folder / f"registration_params_{apple_nr}.txt")
            
            distances = np.linalg.norm(photo_points-ct_points, axis=1) * scale * voxel_size
            per_apple_distances[apple_nr] = distances
            all_distances += list(distances)
        except:
            print(f"Problem with slice {(apple_nr, photo_nr)}")
            continue
        
    np_dist = np.array(all_distances)
    mean = np.mean(np_dist)
    std = np.std(np_dist, ddof=1)
    print(f"Mean = {mean}, std = {std}")
    
    num_usable_slices = []
    min_max_slice_distances = []
    max_slice_distances = []
    mean_errors = []
    
    for apple_nr, selected_slice in selected_slices:
        print(apple_nr)
        scan_name = str(apple_nr)
        photo_path = scans_folder / "slice_photos_crop" / scan_name
        unusable_slices = get_unusable_slices(apple_nr, photo_metadata)
        if unusable_slices is None:
            continue
        usable_slices = get_usable_slices(photo_path, scan_name, unusable_slices)
        if not apple_nr in per_apple_distances:
            print(f"Missing annotations on scan {apple_nr}")
            continue
        num_usable_slices.append(len(usable_slices))
        min_max_slice_distances.append(min(abs(unusable_slices[0]-selected_slice), abs(unusable_slices[-1]-selected_slice)))
        max_slice_distances.append(max(abs(unusable_slices[0]-selected_slice), abs(unusable_slices[-1]-selected_slice)))
        mean_errors.append(np.mean(per_apple_distances[apple_nr]))
        
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(min_max_slice_distances, mean_errors)
    print(f"min_max_slice_distances: slope={slope}, intercept={intercept}, r_value={r_value}, p_value={p_value}, std_err={std_err}")
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(max_slice_distances, mean_errors)
    print(f"max_slice_distances: slope={slope}, intercept={intercept}, r_value={r_value}, p_value={p_value}, std_err={std_err}")
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(num_usable_slices, mean_errors)
    print(f"num_usable_slices: slope={slope}, intercept={intercept}, r_value={r_value}, p_value={p_value}, std_err={std_err}")
    
    plt.scatter(max_slice_distances, mean_errors)
    plt.show()
        
