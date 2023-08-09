import numpy as np
import skimage
import skimage.measure
import skimage.filters
import skimage.morphology
import skimage.io
import matplotlib.pyplot as plt
import matplotlib
import tifffile
from pathlib import Path

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder

class MarkCTCoreCornersPlot():
    def __init__(self, photo, ct, offset, photo_endpoints, out_path):
        self.photo = photo
        self.ct = ct
        self.offset = offset
        self.out_path = out_path
        
        self.photo_endpoints = [(p[0]-offset[0], p[1]-offset[1]) for p in photo_endpoints]
        self.ct_endpoints = []
        
        self.fig, (self.ax_photo, self.ax_ct) = plt.subplots(1, 2, figsize=(20, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.redraw_photo()
        self.redraw_ct()
        plt.tight_layout()
        plt.show()
    
    
    def redraw_photo(self):
        self.ax_photo.clear()
        self.ax_photo.imshow(self.photo)
        #self.ax.set_title(f"Slice {self.curr_slice}")
        self.ax_photo.scatter(
            [point[0] for point in self.photo_endpoints],
            [point[1] for point in self.photo_endpoints],
            marker="x"
            )
        for i, point in enumerate(self.photo_endpoints):
            self.ax_photo.annotate(str(i), (point[0]+5, point[1]-5))
        self.fig.canvas.draw()    
        
    def redraw_ct(self):
        self.ax_ct.clear()
        self.ax_ct.imshow(self.ct)
        #self.ax.set_title(f"Slice {self.curr_slice}")
        self.ax_ct.scatter(
            [point[0] for point in self.ct_endpoints],
            [point[1] for point in self.ct_endpoints],
            marker="x",
            c="r"
            )
        for i, point in enumerate(self.ct_endpoints):
            self.ax_ct.annotate(str(i), (point[0]+5, point[1]-5), c="r")
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == "u":
            self.ct_endpoints.pop()
            self.redraw_ct()
        if event.key in [str(i) for i in range(10)]:
            self.ct_endpoints.pop(int(event.key))
            self.redraw_ct()
            
            
    def on_click(self, event):
        if event.xdata != None and event.ydata != None:
            rx = round(event.xdata)
            ry = round(event.ydata)
            
            if event.button == matplotlib.backend_bases.MouseButton.LEFT:
                self.ct_endpoints.append((rx, ry))
            self.redraw_ct()
            
    def on_close(self, event):
        self.write()
        
    def write(self):
        with open(self.out_path, "w") as out_file:
            for point in self.ct_endpoints:
                out_file.write(f"{point[0]+self.offset[0]},{point[1]+self.offset[1]}\n")

def read_tuple_csv(csv_path):
    with open(csv_path, "r") as csv_file:
        lines = csv_file.readlines()
    return [tuple([int(x) for x in line.split(",")]) for line in lines]

if __name__ == "__main__":
    scans_folder = get_data_folder()
    csv_path = scans_folder / "selected_point_compare_slice.csv"
    
    selected_slices = read_tuple_csv(csv_path)
    
    results_folder = get_results_folder()
    experiment_folder = ceu.make_new_experiment_folder(results_folder, name="click_ct_core_endpoints_overlap")
    #experiment_folder = Path("/home/dirkschut/Experiments/experiments_kanzi_apple_browning/2023-03-07_click_ct_core_endpoints_1")
    ct_experiment_folder = results_folder / "2023-03-24_calc_kanzi_ct_slices_2"
    points_folder = experiment_folder / "annotated_points"
    points_folder.mkdir(exist_ok=True)
    
    for apple_nr, photo_nr in selected_slices:
        photo = skimage.io.imread(scans_folder / "slice_photos_crop" / f"{apple_nr}" / f"Kanzi{apple_nr}_slice_{photo_nr}.png")
        annotations_path = results_folder / "2023-03-05_click_core_endpoints_4" / "annotated_points" / f"apple_{apple_nr}_photo_{photo_nr}.csv"
        try:
            photo_endpoints = read_tuple_csv(annotations_path)
            #print(f"Opening apple {apple_nr}, slice {photo_nr}")
        except:
            print(f"  Error opening apple {apple_nr}, slice {photo_nr}")
        ct_path = ct_experiment_folder / "registered_ct_slices" / f"{apple_nr}" / f"Kanzi{apple_nr}_slice_{photo_nr}.tiff"
        if not ct_path.exists():
            print(f"  Error opening apple {apple_nr}, slice {photo_nr}")
        #if (points_folder / f"apple_{apple_nr}_photo_{photo_nr}.csv").exists() or not ct_path.exists():
        #    continue
        ct = np.flip(tifffile.imread(ct_path), axis=1)
        corner_clicker = MarkCTCoreCornersPlot(
            photo[400:-400,400:-400,:],
            ct[400:-400,400:-400],
            (400,400),
            photo_endpoints,
            points_folder / f"apple_{apple_nr}_photo_{photo_nr}.csv"
            )
