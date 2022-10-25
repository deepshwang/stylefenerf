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
  'curly_hair': ['curly_hair', False, False, True], 
  'Hilary_Clinton': ['hilary_clinton', False, False, False],
  'Jhonny_Depp': ['depp', False, False, False], 
  'mohawk': ['mohawk', False, False, True],
  'purple_hair': ['purple_hair', False, False, False], 
  'surprised': ['surprised', False, False, True], 
  'Taylor_Swift': ['taylor_swift', False, False, False],
  'trump': ['trump', False, False, False], 
  'Mark_Zuckerberg': ['zuckerberg', False, False, False]    
}

trained_meta_data = {
                  "arched_eyebrows": ['arched_eyebrows', False, False, True],  
                  "bald": ['bald', False, False, True], 
                  "beard": ['bald', False, False, False], 
                  "bangs": ['bangs', False, False, True],
                  "big_lips": ['big_lips', False, False, True], 
                  "big_nose": ['big_nose', False, False, True], 
                  "blue_nose": ['blue_nose', False, False, False],
                  "brown_hair": ['brown_hair', False, False, False],  
                  "bushy_eyebrows": ['bushy_eyebrows', False, False, True],
                  "closed_eyes": ['closed_eyes', False, False, True], 
                  "disgusted": ['disgusted', False, False, True],
                  "elf_ear": ['elf_ear', False, False, True], 
                  "eyeglasses": ['eyeglasses', False, False, True],
                  "eyeglasses2": ['eyeglasses', False, False, True],
                  "goatee": ['goatee', False, False, False],
                  "green_lips": ['green_lips', False, False, False],
                  "grey_hair": ['grey_hair', False, False, False], 
                  "happy": ['happy', False, False, True], 
                  "mustache": ['mustache', False, False, False], 
                  "open_mouth": ['open_mouth', False, False, True], 
                  "pale_skin": ['pale_skin', False, False, False],
                  "purple_nose": ['purple_nose', False, False, False], 
                  "sad": ['sad', False, False, True], 
                  "smiling": ['smiling', False, False, True], 
                  "straight_hair": ['straight_hair', False, False, True], 
                  "yellow_lips": ['yellow_lips', False, False, False]
                }


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_type", type=str, default='surprised')
    parser.add_argument("--latent_path", type=str, default="/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_latents.pt")
    parser.add_argument("--exp_dir", type=str, default="/home/nas4_user/sungwonhwang/logs/FENeRF/render/StyleCLIP_edit") 
    args = parser.parse_args()
    
    edit_type = args.edit_type
    model_src = None
    if edit_type in list(trained_meta_data.keys()):
      model_src = "trained"
    elif edit_type in list(meta_data.keys()):
      model_src = "pretrained"
    exp_dir = args.exp_dir + "/{}/".format(edit_type)
    latent_path = args.latent_path
    ensure_checkpoint_exists(latent_path)
    if model_src == "pretrained":
      # if "_" in edit_type:
      #   edit_type = " ".join(edit_type.split("_"))
      edit_id = meta_data[edit_type][0]
      os.makedirs("mapper/pretrained", exist_ok=True)
      ensure_checkpoint_exists(f"mapper/pretrained/{edit_id}.pt")
      checkpoint_path = f"mapper/pretrained/{edit_id}.pt"

    elif model_src == "trained":
      checkpoint_path = "/home/nas1_userD/daejinkim/works/3D_edit/StyleCLIP/styleclip_models/{}/best_model.pt".format(edit_type)
      meta_data = trained_meta_data

    args = {
        "work_in_stylespace": False,
        "exp_dir": exp_dir,
        "checkpoint_path": checkpoint_path,
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

