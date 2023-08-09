import numpy as np
import torch
from tqdm import tqdm
    
class MomentumOptimizer():
    def __init__(self, param_init_values, learning_rates, momentums):
        self.params = param_init_values
        self.dtype = param_init_values[0].dtype
        self.device = param_init_values[0].device
        self.steps = [torch.zeros_like(param, requires_grad=False) for param in self.params]
        self.learning_rates = self._prepare_list(learning_rates)
        self.momentums = self._prepare_list(momentums)
        
    def _prepare_list(self, li):
        result = []
        for el in li:
            if not isinstance(el, torch.Tensor):
                result.append(torch.tensor(el, dtype=self.dtype, device=self.device))
            else:
                result.append(el.to(dtype=self.dtype, device=self.device))
        return result
        
    def step(self):
        with torch.no_grad():
            for param, step, learning_rate, momentum in zip(self.params, self.steps, self.learning_rates, self.momentums):
                step[...] = param.grad * learning_rate + momentum * step
                param -= step
                param.grad.zero_()

class PhotoCtRegistration():
    def __init__(self, tm):
        self.tm = tm
        self.photo_nrs = tm.slice_nrs
        
    def _make_profile(self, mask):
        slice_area = torch.sum(mask.to(dtype=torch.float64), dim=(1, 2))
        profile = torch.sqrt(slice_area)
        return profile
    
    def initialize_on_profiles(self, ct_mask, photo_masks, num_iterations):
        slice_thickness_lr = 100
        slice_offset_lr = 10000
        scale_lr = 0.1
        momentum = 0.6
        
        ct_profile = self._make_profile(ct_mask)
        photo_profile = self._make_profile(torch.from_numpy(np.stack(photo_masks)))
        
        slice_thickness = torch.tensor((ct_profile.size()[0]/(2*(max(self.photo_nrs)-min(self.photo_nrs)))),
            dtype=torch.float32)
        slice_thickness.requires_grad_()
        photo_nr_max_diameter = self.photo_nrs[torch.argmax(photo_profile)]
        slice_offset = (torch.argmax(ct_profile)-(-photo_nr_max_diameter)*slice_thickness.detach()) \
            .to(dtype=torch.float32).requires_grad_(True)
        scale = (torch.max(ct_profile)/torch.max(photo_profile)) \
            .to(dtype=torch.float32).requires_grad_(True)
        
        params = [slice_thickness, slice_offset, scale]
        learning_rates = [slice_thickness_lr, slice_offset_lr, scale_lr]
            
        optimizer = MomentumOptimizer(params, learning_rates, [momentum]*len(params))
        torch_photo_nrs = torch.tensor(self.photo_nrs)

        for i in tqdm(range(num_iterations)):
            transformed_photo_indices = (-torch_photo_nrs*slice_thickness + slice_offset)
            x_diff = transformed_photo_indices[None, :] - torch.arange(0, len(ct_profile), dtype=torch.float32)[:, None]
            y_diff = photo_profile[None, :] * scale - ct_profile[:, None]
            sq_dist = (x_diff/1000)**2 + (y_diff/1000)**2
            
            minvalues, _ = torch.min(sq_dist, dim=0)
            mse = torch.mean(minvalues)
            mse.backward()
            #print(f"MSE = {mse.detach().cpu()}")
            
            optimizer.step()
            
        self.tm.slice_thickness = slice_thickness.detach()
        self.tm.slice_offset = slice_offset.detach()
        self.tm.scale = scale.detach()
        
        return mse.detach(), ct_profile, photo_profile
        
    def initialize_xy_offsets(self, ct_mask, photo_masks):
        for photo_nr, photo_mask in zip(self.photo_nrs, photo_masks):
            photo_center_of_mass = torch.from_numpy(np.mean(np.argwhere(photo_mask).astype(np.float32), axis=0))
            photo_center_of_mass *= self.tm.scale
            
            ct_slice = round(float(-photo_nr*self.tm.slice_thickness + self.tm.slice_offset))
            ct_center_of_mass = torch.mean(torch.argwhere(ct_mask[ct_slice,:,:]).to(dtype=torch.float32), axis=0)
            
            self.tm.slice_xy_offsets[photo_nr] = torch.flipud(ct_center_of_mass[0:2] - photo_center_of_mass)

    def optimize_cost_function(self, cost_function, learning_rates, momentum, max_iterations, device, error_log_path, stopping_condition=None, verbose=False):
        self.tm.to_(device)
        self.tm.require_grad_()
        cost_function.to_(device)
        
        optimizer = MomentumOptimizer(self.tm.get_differentiable_params(), learning_rates, [momentum]*len(learning_rates))
        
        cost_log = []
        
        with open(error_log_path, "w") as error_log_file:            
            for iter_num in (pbar := tqdm(range(max_iterations))):
                cost = cost_function.evaluate(self.tm)
                cost.backward()
                cost_log.append(cost.detach().cpu().item())
                pbar.set_postfix(cost=f"{cost_log[-1]:.6f}")
                
                if stopping_condition is not None and stopping_condition(cost_log):
                    break
                
                if verbose:
                    self.tm.print_with_grad()
                        
                error_log_file.write(f"{cost_log[-1]}\n")
                optimizer.step()
            
        self.tm.detach_()
        self.tm.to_("cpu")
            
        return cost.detach().cpu()
