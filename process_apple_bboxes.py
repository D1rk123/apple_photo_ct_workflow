from folder_locations import get_data_folder
import numpy as np
import skimage.io

class BboxDataParser:
    def __init__(self, bbox_file_path, full_size, edge_clamp_range):
        self.full_size = full_size
        self.edge_clamp_range = edge_clamp_range
        self.bbox_data = np.loadtxt(bbox_file_path, delimiter=',', dtype=int)
        self.clamp_to_edges()
        self.calc_crop_size()
        
    def calc_crop_size(self):
        bbox_widths = self.bbox_data[:, 5] - self.bbox_data[:, 4] + 1
        bbox_heights = self.bbox_data[:, 3] - self.bbox_data[:, 2] + 1
        self.crop_size = (np.max(bbox_heights), np.max(bbox_widths))
        
    def clamp_to_edges(self):
        for bbox in self.bbox_data:
            if bbox[2] < self.edge_clamp_range:
                bbox[2] = 0
            if bbox[3] > self.full_size[0]-self.edge_clamp_range:
                bbox[3] = self.full_size[0]
            if bbox[4] < self.edge_clamp_range:
                bbox[4] = 0
            if bbox[5] > self.full_size[1]-self.edge_clamp_range:
                bbox[5] = self.full_size[1]
                
    def get_crop_info_from_index(self, i):
        return CropInfo(self.bbox_data[i], self.crop_size, self.full_size)
        
    def get_crop_info(self, scan_nr, photo_nr):
        return CropInfo(self.bbox_data[(self.bbox_data[:,0]==scan_nr)&(self.bbox_data[:,1]==photo_nr)][0, :], self.crop_size, self.full_size)
        
    def get_photo_indices(self, i):
        return self.bbox_data[i, 0], self.bbox_data[i, 1]
        
    def __len__(self):
        return len(self.bbox_data)

class CropInfo:
    def __init__(self, bbox, crop_size, full_size):
        self.crop_size = np.array(crop_size, dtype=int)
        self.full_size = np.array(full_size, dtype=int)
        self.bbox_mid = np.array([(bbox[3] + bbox[2])/2, (bbox[5] + bbox[4])/2], dtype=float)
        
        self.crop_min = np.floor(self.bbox_mid - self.crop_size.astype(float)/2).astype(int)
        self.crop_max = self.crop_min + self.crop_size
        
        self.overflow_min = np.maximum(0, -self.crop_min)
        self.overflow_max = np.maximum(0, self.crop_max-self.full_size)
        #print(self.crop_size)
        #print(self.crop_min)
        #print(self.crop_max)
        #print(self.overflow_min)
        #print(self.overflow_max)
        
    def get_full_to_crop_extents(self):
        return self.crop_min+self.overflow_min, self.crop_max-self.overflow_max
        
    def get_crop_content_extents(self):
        return self.overflow_min, self.crop_size-self.overflow_max


def crop_around_bbox(in_path, out_path, crop_info, background_color=None):
    img_in = skimage.io.imread(in_path)
    img_out = np.zeros((crop_info.crop_size[0], crop_info.crop_size[1], 3), dtype=np.uint8)
    if background_color is not None:
        img_out[:, :, :] = background_color[None, None, :]
    
    in_min, in_max = crop_info.get_full_to_crop_extents()
    out_min, out_max = crop_info.get_crop_content_extents()
    
    img_out[out_min[0]:out_max[0], out_min[1]:out_max[1], 0:3] \
        = img_in[in_min[0]:in_max[0], in_min[1]:in_max[1], 0:3]
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(out_path, img_out)

if __name__ == "__main__":
    scans_folder = get_data_folder()
    photos_path = scans_folder / "slice_photos"
    masks_path = scans_folder / "slice_photo_masks"
    photos_crop_path = scans_folder / "slice_photos_crop"
    masks_crop_path = scans_folder / "slice_photo_masks_crop_2"
    bbox_file_path = photos_path / "photo_bboxes.csv"
    full_size = (1942, 2590)
    
    bbox_data = BboxDataParser(bbox_file_path, full_size, 50)


    """
    for i in range(len(bbox_data)):
        print(bbox_data[i,0], bbox_data[i,1])
        in_path = photos_path / f"{bbox_data[i,0]}" / f"Kanzi{bbox_data[i,0]}_slice_{bbox_data[i,1]}.png"
        out_path = photos_crop_path / f"{bbox_data[i,0]}" / f"Kanzi{bbox_data[i,0]}_slice_{bbox_data[i,1]}.png"
        crop_around_bbox(in_path, out_path, bbox_data[i, :], full_res, max_height, max_width)
    """
        
    for i in range(len(bbox_data)):
        photo_indices = bbox_data.get_photo_indices(i)
        print(photo_indices)
        folder_name = f"{photo_indices[0]}"
        file_name = f"Kanzi{photo_indices[0]}_slice_{photo_indices[1]}.png"
        in_path = masks_path / folder_name / file_name
        out_path = masks_crop_path / folder_name / file_name
        
        if in_path.exists():
            crop_info = bbox_data.get_crop_info_from_index(i)
            crop_around_bbox(in_path, out_path, crop_info, np.array((255, 255, 255)))
        

