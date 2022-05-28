#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import tarfile

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import torch

sys.path.insert(0, 'face_detection')
sys.path.insert(0, 'face_alignment')

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

REPO_URL = 'https://github.com/ibug-group/face_alignment'
TITLE = 'ibug-group/face_alignment'
DESCRIPTION = f'This is a demo for {REPO_URL}.'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['001.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def load_detector(device: torch.device) -> RetinaFacePredictor:
    model = RetinaFacePredictor(
        threshold=0.8,
        device=device,
        model=RetinaFacePredictor.get_model('mobilenet0.25'))
    return model


def load_model(model_name: str, device: torch.device) -> FANPredictor:
    model = FANPredictor(device=device,
                         model=FANPredictor.get_model(model_name))
    return model


def predict(image: np.ndarray, model_name: str, max_num_faces: int,
            landmark_score_threshold: int, detector: RetinaFacePredictor,
            models: dict[str, FANPredictor]) -> np.ndarray:
    model = models[model_name]

    # RGB -> BGR
    image = image[:, :, ::-1]

    faces = detector(image, rgb=False)
    if len(faces) == 0:
        raise RuntimeError('No face was found.')
    faces = sorted(list(faces), key=lambda x: -x[4])[:max_num_faces]
    faces = np.asarray(faces)
    landmarks, landmark_scores = model(image, faces, rgb=False)

    res = image.copy()
    for face, pts, scores in zip(faces, landmarks, landmark_scores):
        box = np.round(face[:4]).astype(int)
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
        for pt, score in zip(np.round(pts).astype(int), scores):
            if score < landmark_score_threshold:
                continue
            cv2.circle(res, tuple(pt), 2, (0, 255, 0), cv2.FILLED)

    return res[:, :, ::-1]


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    detector = load_detector(device)

    model_names = [
        '2dfan2',
        '2dfan4',
        '2dfan2_alt',
    ]
    models = {name: load_model(name, device=device) for name in model_names}

    func = functools.partial(predict, detector=detector, models=models)
    func = functools.update_wrapper(func, predict)

    image_paths = load_sample_images()
    examples = [[path.as_posix(), model_names[0], 10, 0.2]
                for path in image_paths]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(model_names,
                            type='value',
                            default=model_names[0],
                            label='Model'),
            gr.inputs.Slider(
                1, 20, step=1, default=10, label='Max Number of Faces'),
            gr.inputs.Slider(
                0, 1, step=0.05, default=0.2,
                label='Landmark Score Threshold'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
