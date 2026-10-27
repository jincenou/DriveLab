"""Turn a video or Waymo TFRecord into the shared, RGB-first scene contract."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import struct
from pathlib import Path

from .io_utils import write_json


def scene_key(path: Path) -> str:
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', path.stem)[:64]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return f'{stem}_{digest.hexdigest()[:12]}'


async def download(url: str, directory: Path) -> Path:
    """Use yt-dlp's supported sites; never execute URL text through a shell."""
    import sys
    if not url.startswith(('https://', 'http://')):
        raise ValueError('Expected an HTTP(S) video URL')
    directory.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'yt_dlp', '--no-playlist', '--no-progress',
        '--restrict-filenames', '--merge-output-format', 'mp4',
        '--print', 'after_move:filepath', '-o', str(directory / '%(id)s.%(ext)s'),
        '--', url, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await process.communicate()
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
    if process.returncode:
        raise RuntimeError('Video download failed: ' + err.decode(errors='replace')[-3000:])
    paths = [Path(line).resolve() for line in out.decode().splitlines() if line.strip()]
    if len(paths) != 1 or not paths[0].is_file() or not paths[0].is_relative_to(directory.resolve()):
        raise ValueError('Downloader did not return exactly one local video')
    return paths[0]


def video(path: Path, output: Path, *, sample_hz=2.0) -> Path:
    import cv2
    if sample_hz <= 0:
        raise ValueError('sample_hz must be positive')
    source = scene_key(path)
    images = output / 'images' / source
    images.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Video has no usable frame rate')
    frames, pending = [], []
    native, next_time, previous = 0, 0.0, -1.0
    try:
        while True:
            ok, rgb = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if native and timestamp <= previous:
                raise ValueError('Non-monotonic decoder timestamps; normalize the video with FFmpeg first')
            previous = timestamp
            image = images / f'{native:06d}.jpg'
            if not cv2.imwrite(str(image), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(image)
            if timestamp + 1e-6 >= next_time:
                frames.append(dict(path=str(image.resolve()), camera_name='front',
                    frame_id=len(frames), native_frame_index=native,
                    timestamp_us=round(timestamp * 1e6), motion_paths=pending))
                pending = []
                next_time = timestamp + 1 / sample_hz
            else:
                pending.append(str(image.resolve()))
            native += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError('Video contains no decodable frames')
    index = output / 'indexes' / f'{source}.json'
    write_json(index, dict(scene_id=source, dataset='video', source_video=str(path.resolve()),
        timestamp_basis='decoder presentation timestamps; relative to video start',
        calibration_available=False, frames=frames, native_frames=native, detector_rate_hz=sample_hz))
    return index


def records(path: Path):
    """Read uncompressed TFRecord framing without importing TensorFlow."""
    import google_crc32c
    def masked(data):
        crc = google_crc32c.value(data)
        return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff
    with path.open('rb') as stream:
        while header := stream.read(12):
            if len(header) != 12 or masked(header[:8]) != struct.unpack('<I', header[8:])[0]:
                raise ValueError('Invalid TFRecord header or checksum')
            size = struct.unpack('<Q', header[:8])[0]
            if size > path.stat().st_size - stream.tell() - 4:
                raise ValueError('Truncated TFRecord')
            payload, checksum = stream.read(size), stream.read(4)
            if len(checksum) != 4 or masked(payload) != struct.unpack('<I', checksum)[0]:
                raise ValueError('Invalid TFRecord payload checksum')
            yield payload


def waymo(path: Path, output: Path, *, stride=5, cameras=None) -> Path:
    import cv2
    import numpy as np
    from waymo_open_dataset import dataset_pb2
    if stride < 1:
        raise ValueError('Waymo stride must be positive')
    names = {1:'front', 2:'front_left', 3:'front_right', 4:'side_left', 5:'side_right'}
    chosen = set(cameras or names.values())
    if not chosen <= set(names.values()):
        raise ValueError('Unknown Waymo camera')
    source = scene_key(path)
    images = output / 'images' / source
    images.mkdir(parents=True, exist_ok=True)
    native_from_cv = np.eye(4)
    native_from_cv[:3, :3] = [[0,0,1], [-1,0,0], [0,-1,0]]
    maps, streams, frames = {}, {c:[] for c in names}, []
    for index, data in enumerate(records(path)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(data)
        calibrations = {c.name:c for c in frame.context.camera_calibrations}
        for image in frame.images:
            if names[image.name] not in chosen:
                continue
            cal = calibrations[image.name]
            intrinsic = np.asarray(cal.intrinsic)
            k = np.array([[intrinsic[0],0,intrinsic[2]], [0,intrinsic[1],intrinsic[3]], [0,0,1.]])
            if image.name not in maps:
                maps[image.name] = cv2.initUndistortRectifyMap(k, intrinsic[4:], None, k,
                    (cal.width, cal.height), cv2.CV_32FC1)
            rgb = cv2.imdecode(np.frombuffer(image.image, np.uint8), cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2] != (cal.height, cal.width):
                raise ValueError('Invalid Waymo image')
            rgb = cv2.remap(rgb, *maps[image.name], cv2.INTER_LINEAR)
            target = images / f'f{index:06d}_c{image.name}.jpg'
            if not cv2.imwrite(str(target), rgb, [cv2.IMWRITE_JPEG_QUALITY,95]):
                raise IOError(target)
            history = streams[image.name]
            if index % stride == 0:
                pose = np.array(image.pose.transform).reshape(4,4)
                c2w = pose @ np.array(cal.extrinsic.transform).reshape(4,4) @ native_from_cv
                frames.append(dict(path=str(target.resolve()), camera_name=names[image.name],
                    frame_id=index//stride, native_frame_index=index,
                    timestamp_us=round(image.pose_timestamp*1e6), camera_intrinsic=k.tolist(),
                    camera_to_world=c2w.tolist(), motion_paths=history[-(stride-1):] if stride>1 else []))
            history.append(str(target.resolve()))
    if not frames:
        raise ValueError('No selected camera frames in TFRecord')
    result = output / 'indexes' / f'{source}.json'
    write_json(result, dict(scene_id=source, dataset='waymo', source_tfrecord=str(path.resolve()),
        rectified=True, annotation_use='none; RGB/calibration only', frames=frames))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', choices=['video','waymo'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-hz', type=float, default=2)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--cameras', nargs='+')
    args = parser.parse_args()
    result = video(args.input, args.output, sample_hz=args.sample_hz) if args.type == 'video' else waymo(
        args.input, args.output, stride=args.stride, cameras=args.cameras)
    print(json.dumps({'index':str(result)}))
