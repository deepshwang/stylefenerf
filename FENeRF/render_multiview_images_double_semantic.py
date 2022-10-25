import argparse
import math
import glob
import numpy as np
import sys
import os
from train_double_latent_semantic import mask2color
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import curriculums
import ipdb
import random
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()
    
def generate_img(gen, z_geo, z_app, **kwargs):
    
    with torch.no_grad():
        img, depth_map = generator.staged_forward(z_geo, z_app, **kwargs)
        img, segmap = img[:, -3:], img[:, :-3]
    return img, mask2color(segmap) / 255.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    parser.add_argument('--create_real_for_fid', type=bool, default=False)

    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    # curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['last_back'] = False
    curriculum['nerf_noise'] = 0
    # curriculum['fill_mode'] = 'weight'
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.output_dir+"/RGB_orig", exist_ok=True)
    os.makedirs(opt.output_dir+"/RGB_real", exist_ok=True)
    os.makedirs(opt.output_dir+"/SEG_orig", exist_ok=True)

    generator = torch.load(opt.path, map_location=torch.device(device))
    generator.softmax_label = False
    generator.neural_renderer_img = None
    generator.neural_renderer_seg = None
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters()) # TODO: what is ema? why use ema parameter?
    generator.set_device(device)
    generator.eval()
    
    face_angles = [-0.5, -0.25, 0., 0.25, 0.5]
    # face_angles = [-0.5, 0., 0.5]

    face_angles = [a + math.pi/2 for a in face_angles]
    used_face_angles = []
    os.makedirs(os.path.join(opt.output_dir, 'fenerf_latents'), exist_ok=True)
    for seed in tqdm(range(opt.seeds)):
        seed = int(seed)
        torch.manual_seed(seed)
        z_geo = torch.randn((1, 256), device=device)
        z_app = torch.randn((1, 256), device=device)
        rand_angle_idx = random.randint(0,len(face_angles)-1) 

        # Saving latents
        geo_frequencies, geo_phase_shifts = generator.siren.geo_mapping_network(z_geo)
        geo_frequency_offsets = torch.zeros_like(geo_frequencies)
        geo_phase_shift_offsets = torch.zeros_like(geo_phase_shifts)
        
        app_frequencies, app_phase_shifts = generator.siren.app_mapping_network(z_app)
        app_frequency_offsets = torch.zeros_like(app_frequencies)
        app_phase_shift_offsets = torch.zeros_like(app_phase_shifts)

        meta = {
                'w_geo_frequencies': geo_frequencies,
                'w_geo_phase_shifts': geo_phase_shifts,
                'w_geo_frequency_offsets': geo_frequency_offsets,
                'w_geo_phase_shift_offsets': geo_phase_shift_offsets,
                'w_app_frequencies': app_frequencies,
                'w_app_phase_shifts': app_phase_shifts,
                'w_app_frequency_offsets': app_frequency_offsets,
                'w_app_phase_shift_offsets': app_phase_shift_offsets
        }
        
        #if seed < opt.seeds:
        if True:
            torch.save(meta, os.path.join(opt.output_dir, 'fenerf_latents', 'freq_phase_offset_{}.pt'.format(str(seed).zfill(5))) ) 
            
        
            # Render with saved latents
            used_face_angles.append(face_angles[rand_angle_idx])
            curriculum['h_mean'] = face_angles[rand_angle_idx]
            img, segmap = generate_img(generator, z_geo, z_app, **curriculum)
            save_image(img, os.path.join(opt.output_dir, 'RGB_orig/{}.png'.format(str(seed).zfill(5))), normalize=True, range=(-1,1))
        #else:
        #    curriculum['h_mean'] = face_angles[rand_angle_idx]
        #    img, segmap = generate_img(generator, z_geo, z_app, **curriculum)
        #    save_image(img, os.path.join(opt.output_dir, 'RGB_real/{}.png'.format(str(seed-opt.seeds).zfill(5))), normalize=True, range=(-1,1))
 
        #save_image(segmap, os.path.join(opt.output_dir, 'SEG_orig/{}.png'.format(str(seed).zfill(5))), noralize=True, range=(0,1))
    
    with open(opt.output_dir + '/face_angles_{}.pkl'.format(opt.seeds), 'wb') as f:
        pickle.dump(used_face_angles, f) 
