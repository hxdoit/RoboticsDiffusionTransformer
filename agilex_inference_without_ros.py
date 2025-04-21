#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import yaml
from collections import deque

import numpy as np

import torch

from PIL import Image as PImage

from scripts.agilex_model import create_model


CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']


# Initialize the model
def make_policy():
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained="robotics-diffusion-transformer/rdt-1b",
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=30,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)



def get_config():
    config = {

        'state_dim': 14,
        'chunk_size': 64,
        'camera_names': CAMERA_NAMES,
    }
    return config




# RDT inference
def inference_fn(config, policy):
    observation_window = deque(maxlen=2)

    # Append the first dummy image
    observation_window.append(
        {
            'qpos': torch.zeros(14),
            'images':
                {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
        }
    )
    observation_window.append(
        {
            'qpos': torch.zeros(14),
            'images':
                {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
        }
    )
    lang_dict = torch.load("/home/ubuntu/Downloads/rdt/RoboticsDiffusionTransformer/outs/handover_pan.pt")
    lang_embeddings = lang_dict.unsqueeze(0)

    image_arrs = [
        observation_window[-2]['images'][config['camera_names'][0]],
        observation_window[-2]['images'][config['camera_names'][1]],
        observation_window[-2]['images'][config['camera_names'][2]],

        observation_window[-1]['images'][config['camera_names'][0]],
        observation_window[-1]['images'][config['camera_names'][1]],
        observation_window[-1]['images'][config['camera_names'][2]]
    ]
    images = [PImage.fromarray(arr) if arr is not None else None
              for arr in image_arrs]


    proprio = observation_window[-1]['qpos']
    # unsqueeze to [1, 14]
    proprio = proprio.unsqueeze(0)

    # actions shaped as [1, 64, 14] in format [left, right]
    actions = policy.step(
        proprio=proprio,
        images=images,
        text_embeds=lang_embeddings
    ).squeeze(0).cpu().numpy()

    return actions


# Main loop for the manipulation task
def model_inference(config):


    # Load rdt model
    policy = make_policy()


    with torch.inference_mode():

        inference_fn(config, policy).copy()








if __name__ == '__main__':
    set_seed(1)
    config = get_config()
    model_inference(config)
