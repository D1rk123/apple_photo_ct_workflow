from folder_locations import get_data_folder
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io

class Bbox_clicker():
    def __init__(self):
        self.num_clicks = 0
        self.min_x = 1e7
        self.max_x = 0
        self.min_y = 1e7
        self.max_y = 0

    def onclick(self, event):
        if event.xdata != None and event.ydata != None:
            self.min_x = min(self.min_x, event.xdata)
            self.min_y = min(self.min_y, event.ydata)
            self.max_x = max(self.max_x, event.xdata)
            self.max_y = max(self.max_y, event.ydata)
            
            self.num_clicks += 1
            if self.num_clicks >= 4:
                plt.close()

def click_bbox(img):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(img)
    clicker = Bbox_clicker()
    cid = fig.canvas.mpl_connect('button_press_event', clicker.onclick)
    plt.show()
    return int(np.floor(clicker.min_y)), int(np.ceil(clicker.max_y)), \
        int(np.floor(clicker.min_x)), int(np.ceil(clicker.max_x))

if __name__ == "__main__":
    scans_folder = get_data_folder()
    photos_path = scans_folder / "slice_photos"
    result_path = photos_path / "photo_bboxes.csv"

    with open(result_path, "w") as result_file:
        for i in range(110, 121):
            for j in range(1, 25):
                file = photos_path / f"{i}" / f"Kanzi{i}_slice_{j}.png"
                if file.exists():
                    img = skimage.io.imread(file)
                    bbox = click_bbox(img)
                    slice_number = re.findall('\d+', file.name)[-1]
                    print(f"({i}, {j}) = {bbox}")
                    result_file.write(f"{i},{j},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")


