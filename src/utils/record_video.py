import os
from gymnasium import logger
from gymnasium.wrappers import RecordVideo


class RecordVideoExtended(RecordVideo):
    """Supports choosing output format and codecs.
    Extends gymnasium's RecordVideo to allow saving videos in WebM and gif formats
    instead of just MP4. WebM is a more efficient format for web distribution.
    """

    ALLOWED_MIME_TYPES = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "gif": "image/gif",
    }

    def __init__(
        self,
        env,
        video_folder,
        episode_trigger,
        name_prefix="rl-video",
        disable_logger=True,
        output_formats: list[str]=["mp4"],
    ):
        """Initialize RecordWebmVideo wrapper.

        Parameters
        ----------
        env : gymnasium.Env
            The environment to wrap
        video_folder : str
            Path to folder where videos will be saved
        episode_trigger : callable
            Function determining which episodes to record
        name_prefix : str, optional
            Prefix for video filenames, by default "rl-video"
        disable_logger : bool, optional
            Whether to disable logging, by default True
        output_formats : list[str], optional
            Video format ("mp4", "webm", or "gif"), by default ["mp4"]
        """
        # Validate output format
        for format in output_formats:
            if format not in self.ALLOWED_MIME_TYPES:
                raise ValueError(
                    f"output_format must be one of {list(self.ALLOWED_MIME_TYPES.keys())}, "
                    f"got {format}"
                )

        self.output_formats = output_formats

        # Call parent constructor
        super().__init__(
            env=env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )

    def stop_recording(self, save: bool = True):
        """Stop current recording and saves the video in the specified format (MP4, WebM, or GIF)."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                from gymnasium import error

                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

        if save:
            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"

            # Build the output path with appropriate extension
            for format in self.output_formats:
                path = os.path.join(
                    self.video_folder, f"{self._video_name}.{format}"
                )

                # Write video in the specified format
                if format == "webm":
                    # Write WebM format with VP9 codec
                    clip.write_videofile(
                        path, codec="libvpx-vp9", audio=False, logger=moviepy_logger
                    )
                elif format == "gif":
                    # Write GIF format
                    clip.write_gif(path, logger=moviepy_logger)
                else:
                    # Default to MP4 format
                    clip.write_videofile(path, codec="libx264", logger=moviepy_logger)

            del clip

        del self.recorded_frames
        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            import gc

            gc.collect()
