import numpy as np
import skimage
import skimage.measure
import skimage.filters
import skimage.morphology
import skimage.io
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder

class MarkCoreCornersPlot():
    def __init__(self, img, offset, out_path, in_path=None):
        self.img = img
        self.offset = offset
        self.out_path = out_path
        
        if in_path is None:
            self.core_endpoints = []
        else:
            self.read(in_path)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.redraw()
        plt.tight_layout()
        plt.show()

        
    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)
        #self.ax.set_title(f"Slice {self.curr_slice}")
        self.ax.scatter(
            [point[0] for point in self.core_endpoints],
            [point[1] for point in self.core_endpoints],
            marker="x"
            )
        for i, point in enumerate(self.core_endpoints):
            self.ax.annotate(str(i), (point[0]+5, point[1]-5))
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == "u":
            self.core_endpoints.pop()
            self.redraw()
        if event.key in [str(i) for i in range(10)]:
            self.core_endpoints.pop(int(event.key))
            self.redraw()
            
            
    def on_click(self, event):
        if event.xdata != None and event.ydata != None:
            rx = round(event.xdata)
            ry = round(event.ydata)
            
            if event.button == matplotlib.backend_bases.MouseButton.LEFT:
                self.core_endpoints.append((rx, ry))
            self.redraw()
            
    def on_close(self, event):
        self.write()
        
    def write(self):
        with open(self.out_path, "w") as out_file:
            for point in self.core_endpoints:
                out_file.write(f"{point[0]+self.offset[0]},{point[1]+self.offset[1]}\n")
    
    def read(self, in_path):
        self.core_endpoints = []
        with open(self.out_path, "r") as in_file:
            for line in in_file.readlines:
                line_parts = line.split(",")
                p0 = int(line_parts[0])-self.offset[0]
                p1 = int(line_parts[1])-self.offset[1]
                self.core_endpoints.append((p0, p1))

def read_slice_csv(csv_path):
    with open(csv_path, "r") as csv_file:
        lines = csv_file.readlines()
    return [tuple([int(x) for x in line.split(",")]) for line in lines]

if __name__ == "__main__":
    scans_folder = get_data_folder()
    csv_path = scans_folder / "selected_point_compare_slice.csv"
    
    selected_slices = read_slice_csv(csv_path)
    
    results_folder = get_results_folder()
    in_points_folder = results_folder / "2023-03-05_click_core_endpoints_4" / "annotated_points"
    experiment_folder = ceu.make_new_experiment_folder(results_folder, name="click_core_endpoints")
    points_folder = experiment_folder / "annotated_points"
    points_folder.mkdir()
    
    for apple_nr, photo_nr in selected_slices:
        img = skimage.io.imread(scans_folder / "slice_photos_crop" / f"{apple_nr}" / f"Kanzi{apple_nr}_slice_{photo_nr}.png")
        corner_clicker = MarkCoreCornersPlot(
            img[400:-400,400:-400,:],
            (400,400),
            points_folder / f"apple_{apple_nr}_photo_{photo_nr}.csv",
            in_points_folder / f"apple_{apple_nr}_photo_{photo_nr}.csv"
        )
