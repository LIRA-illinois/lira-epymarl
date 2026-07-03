from gymnasium.core import ActType
import os
from gymnasium import logger
from gymnasium.wrappers import RecordVideo


class RecordVideoExtended(RecordVideo):
    """Subclass that adds:
    Support for choosing multiple output format and codecs. Allows videos in WebM and gif formats instead of just MP4. WebM is a more efficient format for web distribution.
    State, obs, and avail_actions "getter" methods to interface with the "pre_transition_data" object in the PYMARL training loop.
    Supports hierarchical env in the step() method by rending the high-level action before transitioning the env to the next state
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
        name_prefix: str="rl-video",
        disable_logger: bool=True,
        output_formats: list[str] = ["mp4"],
    ) -> None:
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

    def stop_recording(self, save: bool = True) -> None:
        """Stop current recording single episode and saves the video in the specified format (MP4, WebM, or GIF)."""
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
            for output_format in self.output_formats:
                # manually save video for later logging
                path = os.path.join(
                    self.video_folder, f"{self._video_name}.{output_format}"
                )
                # Write video in the specified format
                if output_format == "webm":
                    # Write WebM format with VP9 codec
                    clip.write_videofile(
                        path, codec="libvpx-vp9", audio=False, logger=moviepy_logger
                    )
                elif output_format == "gif":
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

    def step(self, action: ActType, capture_before_step: bool=True):
        """overrides parent's step(), gives option to capture the frame with the chosen action before the transition occurs
        """
        # render a frame of the env with the chosen action before stepping
        if isinstance(action, dict) and action.get("hl_actions", False):
            self.env.set_wrapper_attr("_pre_step_hl_actions", action["hl_actions"])
            self.env.set_wrapper_attr("_pre_step_actions", action["env_actions"])
        else:
            self.env.set_wrapper_attr("_pre_step_actions", action)

        if capture_before_step:
            if self.recording:
                self._capture_frame()

                if len(self.recorded_frames) > self.video_length:
                    self.stop_recording()

        # code below taken from parent class
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

        if not capture_before_step:
            if self.recording:
                self._capture_frame()

                if len(self.recorded_frames) > self.video_length:
                    self.stop_recording()

        return obs, rew, terminated, truncated, info

    @property
    def state(self):
        return self.env.get_wrapper_attr("state")

    @property
    def obs(self):
        return self.env.get_wrapper_attr("obs")

    @property
    def avail_actions(self):
        return self.env.get_wrapper_attr("avail_actions")
