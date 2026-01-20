import cv2
import torch

from pathlib import Path
from tqdm import tqdm

from config import get_config
from generate_summary import generate_summary
from model import set_model
from video_helper import VideoPreprocessor

def pick_frames(video_path, selections):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    n_frames = 0

    with tqdm(total = len(selections), ncols=90, desc = "selecting frames", unit='frame', leave = False) as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            if selections[n_frames]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            n_frames += 1

            pbar.update(1)
        
    cap.release()

    return frames

def produce_video(save_path, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    for frame in tqdm(frames, total = len(frames), ncols=90, desc = "generating videos", leave = False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def main():
    # Load config
    config = get_config()

    # create output directory
    out_dir = Path(config.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # feature extractor
    video_proc = VideoPreprocessor(
        sample_rate=config.sample_rate,
        device=config.device
    )

    # search all videos with .mp4 suffix
    if config.input_is_file:
        video_paths = [Path(config.file_path)]
    else:
        video_paths = sorted(Path(config.dir_path).glob(f'*.{config.ext}'))


    # Load MARs weights
    model = set_model(
        conformer_model_dim=config.conformer_model_dim,
        conformer_nhead=config.conformer_nhead,
        conformer_num_blocks=config.conformer_num_blocks,
        conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
        conformer_dropout=config.conformer_dropout,
        conformer_order=config.conformer_order
    )
    model.load_state_dict(torch.load(config.weight_path, map_location='cpu'))
    model.to(config.device)
    model.eval()

    # Generate summarized videos
    with torch.no_grad():
        for video_path in tqdm(video_paths,total=len(video_paths),ncols=80,leave=False,desc="Making videos..."):
            video_name = video_path.stem
            n_frames, features, cps, pick = video_proc.run(video_path)

            inputs = features.to(config.device)
            inputs = inputs.unsqueeze(0).expand(3,-1,-1).unsqueeze(0)
            outputs = model(inputs)
            predictions = outputs.squeeze().clone().detach().cpu().numpy().tolist()
            # print(cps.shape, len(predictions), n_frames, pick.shape)
            selections = generate_summary([cps], [predictions], [n_frames], [pick])[0]

            frames = pick_frames(video_path=video_path, selections=selections)
            produce_video(
                save_path=f'{config.save_path}/{video_name}.mp4',
                frames=frames,
                fps=video_proc.fps,
                frame_size=(video_proc.frame_width,video_proc.frame_height)
            )

if __name__=='__main__':
    main()
