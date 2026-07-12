from __future__ import annotations

import os
from pathlib import Path


FFMPEG_BIN = Path(r"C:\ffmpeg-shared\bin")
VIDEO_FILE = Path(r"C:\robot_test_here\ffmpeg_test.mp4")


def main() -> None:
    if not FFMPEG_BIN.is_dir():
        raise FileNotFoundError(
            f"FFmpeg shared-library directory does not exist: {FFMPEG_BIN}"
        )

    if not VIDEO_FILE.is_file():
        raise FileNotFoundError(f"Video file does not exist: {VIDEO_FILE}")

    print("FFmpeg shared directory:", FFMPEG_BIN)
    print("Video file:", VIDEO_FILE)

    dll_handle = os.add_dll_directory(str(FFMPEG_BIN))

    try:
        from torchcodec.decoders import VideoDecoder

        print("TorchCodec imported successfully.")

        decoder = VideoDecoder(str(VIDEO_FILE))

        print("Decoder created successfully.")
        print("Number of frames:", len(decoder))

        frame = decoder[0]

        print("Frame type:", type(frame))
        print("First-frame shape:", tuple(frame.shape))
        print("First-frame dtype:", frame.dtype)
        print("TorchCodec decoding succeeded.")

    finally:
        dll_handle.close()


if __name__ == "__main__":
    main()