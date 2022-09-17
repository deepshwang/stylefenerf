import torch
import os
from utils import ensure_checkpoint_exists
from mapper.scripts.inference import run
from argparse import Namespace
import argparse

meta_data = {
  'afro': ['afro', False, False, True], 
  'angry': ['angry', False, False, True], 
  'Beyonce': ['beyonce', False, False, False], 
  'bobcut': ['bobcut', False, False, True], 
  'bowlcut': ['bowlcut', False, False, True], 
  'curly hair': ['curly_hair', False, False, True], 
  'Hilary Clinton': ['hilary_clinton', False, False, False],
  'Jhonny Depp': ['depp', False, False, False], 
  'mohawk': ['mohawk', False, False, True],
  'purple hair': ['purple_hair', False, False, False], 
  'surprised': ['surprised', False, False, True], 
  'Taylor Swift': ['taylor_swift', False, False, False],
  'trump': ['trump', False, False, False], 
  'Mark Zuckerberg': ['zuckerberg', False, False, False]    
}




if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, default='surprised')
    parser.add_argument("--latent_path", type=str, default="/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_latents.pt")
    parser.add_argument("--exp_dir", type=str, default="/home/nas4_user/sungwonhwang/logs/FENeRF/render/StyleCLIP_edit") 
    args = parser.parse_args()
    
    edit_type = args.edit_type
    exp_dir = args.exp_dir + "/{}/".format(edit_type)
    if "_" in edit_type:
      edit_type = " ".join(edit_type.split("_"))
    edit_id = meta_data[edit_type][0]
    os.makedirs("mapper/pretrained", exist_ok=True)
    ensure_checkpoint_exists(f"mapper/pretrained/{edit_id}.pt")
    latent_path = args.latent_path
    ensure_checkpoint_exists(latent_path)

    args = {
        "work_in_stylespace": False,
        "exp_dir": exp_dir,
        "checkpoint_path": f"mapper/pretrained/{edit_id}.pt",
        "couple_outputs": True,
        "mapper_type": "LevelsMapper",
        "no_coarse_mapper": meta_data[edit_type][1],
        "no_medium_mapper": meta_data[edit_type][2],
        "no_fine_mapper": meta_data[edit_type][3],
        "stylegan_size": 1024,
        "test_batch_size": 1,
        "latents_test_path": latent_path,
        "test_workers": 1,
        "n_images": None 
    }

    run(Namespace(**args))

