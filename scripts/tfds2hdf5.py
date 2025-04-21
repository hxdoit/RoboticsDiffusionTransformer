import cv2
import h5py
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
import os

display_key = 'image'
datasets_name = "cmu_stretch"

b = tfds.builder_from_directory(f"/home/ubuntu/Downloads/openvla/austin_buds_dataset_converted_externally_to_rlds/0.1.0")

ds = b.as_dataset(split='train') # 具体可以根据需求改

output_dir = f'/home/ubuntu/Downloads/rdt/austin_buds_dataset_converted_externally_to_rlds_hdf5/'
os.makedirs(output_dir, exist_ok=True)


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode('.png', imgs[i].numpy())
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        # encode_data.append(np.frombuffer(jpeg_data, dtype='S1'))
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b'\0'))
    return encode_data, max_len

instructions_file_path = os.path.join(output_dir, f'instruction.txt')
state_file_path = os.path.join(output_dir, f'state.txt')
# 遍历数据集
for idx, episode in enumerate(ds):
    # 为每个视频创建一个文件夹
    #video_folder = os.path.join(output_dir, f'video_{idx}')
    #os.makedirs(video_folder, exist_ok=True)

    # 提取该视频的所有帧
    frames = episode['steps']

    # 遍历每一帧并保存
    state_list = []

    # 存储hdf5要使用的数据
    qpos = []
    actions = []
    cam_high = []
    cam_right_wrist = []
    past_state = np.zeros(7)

    for frame_idx, step in enumerate(frames):
        state = step['observation']["state"]
        state = np.array(state)  # x,y,z,rx,ry,rz
        # state = np.append(state, data["gripper"])  # 添加一个张开度0~1
        state = state.astype(np.float32)
        pos = state[:6]
        pos = np.append(pos, state[7])
        qpos.append(pos)
    	# 每个数据集image的特征名字不一样，具体要看数据集下载好后的 features.json 文件中对应的字段是什么
        image = step['observation']['image'] # fractal20220817_data
        wrist_image = step['observation']['wrist_image']  # fractal20220817_data
        # image = step['observation']["agentview_rgb"] # viola
        # image = step['observation']["image"] # bridge

        # 获取自然语言指令，具体要看数据集下载好后的 features.json 文件对应的字段是什么
        natural_language_instruction = step["language_instruction"].numpy().decode('utf-8') # for ucsd、berkeley_fanuc_manipulation
        #natural_language_instruction = step['observation']["natural_language_instruction"].numpy().decode('utf-8')

        #state_list.append(step['observation']["state"])

        # 将图像转换为 PIL 格式
        #image_pil = Image.fromarray(image.numpy())

        # 保存图像，文件名格式为 frame_{frame_idx}.png
        #output_path = os.path.join(video_folder, f'frame_{frame_idx}.png')
        #image_pil.save(output_path)

        if frame_idx == 0:
            pass
        elif frame_idx == len(frames) - 1:
            action = state - past_state
            action_new = action[:6]
            action_new = np.append(action_new, action[7])
            actions.append(action_new)
            actions.append(action_new)  # 最后一次轨迹没有预测，就用最后一次的轨迹本身作为预测
        else:
            action = state - past_state
            action_new = action[:6]
            action_new = np.append(action_new, action[7])
            actions.append(action_new)
        cam_high.append(wrist_image)
        cam_right_wrist.append(image)
        past_state = state

    hdf5path = os.path.join(output_dir, f'episode_{idx}.hdf5')
    with h5py.File(hdf5path, 'w') as f:
        f.create_dataset('action', data=np.array(actions))
        obs = f.create_group('observations')
        image = obs.create_group('images')
        obs.create_dataset('qpos', data=qpos)
        # 图像编码后按顺序存储

        cam_high_enc, len_high = images_encoding(cam_high)
        cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
        image.create_dataset('cam_high', data=cam_high_enc, dtype=f'S{len_high}')
        image.create_dataset('cam_right_wrist', data=cam_right_wrist_enc, dtype=f'S{len_right}')
    with open(instructions_file_path, 'a') as f:
        f.write(f"{natural_language_instruction}\n")

