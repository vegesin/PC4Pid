# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-30
# @FilePath: \SNN\src\scripts\mmradar_viewer.py
# @Description: Viewer for mmWave radar point clouds and paired camera images.
# -------------------------------------------------------

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # yapf: disable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # yapf: disable


try:
    from src.datasets.mmradar_pc import DEFAULT_ANGLE, DEFAULT_HEIGHT, parse_radar_bin
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    DEFAULT_ANGLE = 25.0
    DEFAULT_HEIGHT = 2.0

    def parse_radar_bin(
        bin_path: str,
        install_angle: float = DEFAULT_ANGLE,
        radar_height: float = DEFAULT_HEIGHT,
    ) -> Optional[np.ndarray]:
        """Parses one raw mmWave radar bin frame.

        This fallback mirrors ``src.datasets.mmradar_pc.parse_radar_bin`` so the
        viewer can run without training-only dependencies such as torch.

        Args:
            bin_path: Raw radar bin path.
            install_angle: Radar installation pitch angle in degrees.
            radar_height: Radar installation height in meters.

        Returns:
            Array with columns ``x, y, z, velocity, power_db``.
        """
        try:
            with open(bin_path, "rb") as file:
                frame_data = file.read()
        except Exception as exc:
            print(f"[ERROR] Failed to read bin file: {exc}")
            return None

        if len(frame_data) < 12:
            return None
        if frame_data[0] != 0x55 or frame_data[1] != 0xAA:
            return None

        frame_length = struct.unpack("<I", frame_data[2:6])[0]
        if len(frame_data) < 6 + frame_length:
            return None

        pos = 6
        pos += 2
        pos += 1
        tlv_type = frame_data[pos]
        pos += 1
        if tlv_type != 0x01:
            print(f"[WARN] TLV type {tlv_type} not point cloud, skip")
            return None

        if pos + 2 > len(frame_data):
            return None
        target_num = struct.unpack("<H", frame_data[pos:pos + 2])[0]
        pos += 2

        points = []
        for _ in range(target_num):
            if pos + 9 > len(frame_data):
                break

            idx1 = struct.unpack("<H", frame_data[pos:pos + 2])[0]
            idx2 = frame_data[pos + 2]
            idx3 = frame_data[pos + 3]
            idx4 = frame_data[pos + 4]
            pow_abs = struct.unpack("<I", frame_data[pos + 5:pos + 9])[0]
            pos += 9

            range_val = idx1 * 0.05
            velocity = (idx2 - 32) * 0.104167

            az_arg = idx3 / 64.0 if idx3 <= 63 else (idx3 - 128) / 64.0
            el_arg = idx4 / 64.0 if idx4 <= 63 else (idx4 - 128) / 64.0
            az_rad = math.asin(max(-1.0, min(1.0, az_arg)))
            el_rad = math.asin(max(-1.0, min(1.0, el_arg)))

            x_radar = range_val * math.sin(az_rad) * math.cos(el_rad)
            y_radar = range_val * math.cos(el_rad) * math.cos(az_rad)
            z_radar = range_val * math.sin(el_rad)

            vertical_angle_rad = math.radians(-install_angle)
            z_world = z_radar * math.cos(vertical_angle_rad) + y_radar * math.sin(vertical_angle_rad)
            y_world = -z_radar * math.sin(vertical_angle_rad) + y_radar * math.cos(vertical_angle_rad)
            z_world += radar_height

            power_db = 10 * math.log10(pow_abs + 1e-6) if pow_abs > 0 else -100
            points.append([x_radar, y_world, z_world, velocity, power_db])

        if not points:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(points, dtype=np.float32)


DEFAULT_DATA_ROOT = Path(r"C:\Users\i914900kf\Desktop\毫米波点云数据集\DATA_20260320")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
COLOR_MODES = ("velocity", "snr", "z")


@dataclass(frozen=True)
class FrameItem:
    """Stores one radar frame path and its optional paired image path.

    Args:
        bin_path: Radar raw point-cloud file path.
        image_path: Camera image path with the same stem as ``bin_path``.
    """

    bin_path: Path
    image_path: Optional[Path]


def _find_sequence_dir(initial_dir: Path) -> Optional[Path]:
    """Opens a folder selector for choosing one acquisition sequence.

    Args:
        initial_dir: Initial directory shown by the folder dialog.

    Returns:
        Selected sequence directory, or ``None`` if the dialog is cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        title="Select mmWave acquisition folder",
        initialdir=str(initial_dir) if initial_dir.exists() else str(Path.cwd()),
    )
    root.destroy()
    return Path(selected) if selected else None


def _build_image_index(image_dir: Path) -> dict[str, Path]:
    """Builds a stem-to-image path index for fast radar/image matching.

    Args:
        image_dir: Directory containing camera images.

    Returns:
        Mapping from file stem to image path.
    """
    if not image_dir.exists():
        return {}

    image_index: dict[str, Path] = {}
    for suffix in IMAGE_SUFFIXES:
        for image_path in sorted(image_dir.glob(f"*{suffix}")):
            image_index.setdefault(image_path.stem, image_path)
    return image_index


def load_sequence(sequence_dir: Path) -> list[FrameItem]:
    """Loads radar frame metadata from one acquisition directory.

    Args:
        sequence_dir: Directory containing ``pointcloud`` and ``image`` folders.

    Returns:
        Ordered frame list using point-cloud files as the timeline.

    Raises:
        FileNotFoundError: If the point-cloud folder is missing or empty.
    """
    pointcloud_dir = sequence_dir / "pointcloud"
    image_dir = sequence_dir / "image"

    if not pointcloud_dir.exists():
        raise FileNotFoundError(f"Point-cloud folder not found: {pointcloud_dir}")

    bin_paths = sorted(pointcloud_dir.glob("*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No .bin files found in: {pointcloud_dir}")

    image_index = _build_image_index(image_dir)
    return [FrameItem(bin_path=bin_path, image_path=image_index.get(bin_path.stem)) for bin_path in bin_paths]


class MMRadarViewer:
    """Matplotlib GUI for playback of radar point clouds and paired images."""

    def __init__(
        self,
        frames: list[FrameItem],
        sequence_dir: Path,
        install_angle: float,
        radar_height: float,
        interval_ms: int,
    ) -> None:
        """Initializes the viewer state and widgets.

        Args:
            frames: Ordered frame list.
            sequence_dir: Acquisition sequence directory.
            install_angle: Radar installation pitch angle used by parser.
            radar_height: Radar installation height used by parser.
            interval_ms: Playback interval in milliseconds.
        """
        self.frames = frames
        self.sequence_dir = sequence_dir
        self.install_angle = install_angle
        self.radar_height = radar_height
        self.interval_ms = interval_ms

        self.frame_idx = 0
        self.playing = True
        self.playback_finished = False
        self.color_mode = "velocity"
        self.colorbar = None
        self._suppress_slider = False
        self.cloud_limits = self._compute_global_cloud_limits()

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(f"MMRadar Viewer - {sequence_dir.name}")
        self.ax_cloud = self.fig.add_axes([0.04, 0.20, 0.46, 0.74], projection="3d")
        self.ax_cbar = self.fig.add_axes([0.505, 0.27, 0.012, 0.62])
        self.ax_image = self.fig.add_axes([0.54, 0.27, 0.42, 0.62])
        self.info_text = self.fig.text(
            0.54,
            0.10,
            "",
            fontsize=9,
            family="monospace",
            va="top",
        )
        self.status_text = self.fig.text(
            0.04,
            0.155,
            "",
            color="tab:red",
            fontsize=11,
            weight="bold",
        )

        self._create_widgets()
        self._connect_events()

        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._on_timer)

    def show(self) -> None:
        """Starts playback and displays the GUI."""
        self._draw_frame(0)
        self.timer.start()
        plt.show()

    def _create_widgets(self) -> None:
        """Creates playback buttons, frame slider, and color mode selector."""
        self.ax_prev = self.fig.add_axes([0.04, 0.08, 0.055, 0.045])
        self.ax_play = self.fig.add_axes([0.105, 0.08, 0.07, 0.045])
        self.ax_next = self.fig.add_axes([0.185, 0.08, 0.055, 0.045])
        self.ax_speed_down = self.fig.add_axes([0.25, 0.08, 0.055, 0.045])
        self.ax_speed_up = self.fig.add_axes([0.315, 0.08, 0.055, 0.045])
        self.ax_slider = self.fig.add_axes([0.04, 0.02, 0.46, 0.035])
        self.ax_color = self.fig.add_axes([0.405, 0.075, 0.095, 0.085])

        self.btn_prev = Button(self.ax_prev, "Prev")
        self.btn_play = Button(self.ax_play, "Pause")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_speed_down = Button(self.ax_speed_down, "Slower")
        self.btn_speed_up = Button(self.ax_speed_up, "Faster")
        self.slider = Slider(
            ax=self.ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(self.frames) - 1,
            valinit=0,
            valstep=1,
        )
        self.radio = RadioButtons(self.ax_color, COLOR_MODES, active=0)

        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_play.on_clicked(lambda _: self._toggle_play())
        self.btn_next.on_clicked(lambda _: self._step(1))
        self.btn_speed_down.on_clicked(lambda _: self._change_speed(1.25))
        self.btn_speed_up.on_clicked(lambda _: self._change_speed(0.8))
        self.slider.on_changed(self._on_slider_changed)
        self.radio.on_clicked(self._on_color_changed)

    def _connect_events(self) -> None:
        """Connects keyboard shortcuts."""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    @lru_cache(maxsize=None)
    def _load_points(self, bin_path: str) -> np.ndarray:
        """Loads one radar frame using the shared parser.

        Args:
            bin_path: Radar frame path.

        Returns:
            Parsed point cloud with shape ``(N, 5)``.
        """
        points = parse_radar_bin(
            bin_path,
            install_angle=self.install_angle,
            radar_height=self.radar_height,
        )
        if points is None:
            return np.zeros((0, 5), dtype=np.float32)
        return points

    def _compute_global_cloud_limits(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Computes fixed 3D axis limits from the whole sequence.

        Returns:
            Tuple of ``xlim, ylim, zlim`` limits shared by every frame.
        """
        mins = []
        maxs = []
        print("Scanning radar frames for fixed 3D axis limits...")
        for frame in self.frames:
            points = self._load_points(str(frame.bin_path))
            if len(points) == 0:
                continue
            mins.append(points[:, :3].min(axis=0))
            maxs.append(points[:, :3].max(axis=0))

        if not mins:
            return (-8.0, 8.0), (0.0, 20.0), (-1.0, 5.0)

        xyz_min = np.min(np.stack(mins), axis=0)
        xyz_max = np.max(np.stack(maxs), axis=0)
        centers = (xyz_min + xyz_max) / 2.0
        spans = np.maximum(xyz_max - xyz_min, np.array([2.0, 2.0, 1.0], dtype=np.float32))
        radius = max(float(spans.max()) / 2.0, 1.0)
        margin = max(radius * 0.08, 0.5)
        radius += margin

        xlim = (float(centers[0] - radius), float(centers[0] + radius))
        ylim = (float(centers[1] - radius), float(centers[1] + radius))
        zlim = (float(centers[2] - radius), float(centers[2] + radius))
        print(f"Fixed xlim={xlim}, ylim={ylim}, zlim={zlim}")
        return xlim, ylim, zlim

    @lru_cache(maxsize=64)
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Loads one camera image.

        Args:
            image_path: Image path.

        Returns:
            Image array, or ``None`` if loading fails.
        """
        try:
            return mpimg.imread(image_path)
        except Exception as exc:
            print(f"[WARN] Failed to read image {image_path}: {exc}")
            return None

    def _draw_frame(self, frame_idx: int) -> None:
        """Draws the selected radar frame and paired image.

        Args:
            frame_idx: Frame index to display.
        """
        self.frame_idx = int(np.clip(frame_idx, 0, len(self.frames) - 1))
        if self.frame_idx < len(self.frames) - 1 and self.playback_finished:
            self.playback_finished = False
            self.status_text.set_text("")
        frame = self.frames[self.frame_idx]
        points = self._load_points(str(frame.bin_path))

        self._draw_cloud(points)
        self._draw_image(frame.image_path)
        self._draw_info(frame, points)
        self._sync_slider()
        self.fig.canvas.draw_idle()

    def _draw_cloud(self, points: np.ndarray) -> None:
        """Draws the 3D radar point cloud.

        Args:
            points: Point cloud with columns ``x, y, z, velocity, snr``.
        """
        self.ax_cloud.clear()
        self.ax_cloud.set_title("Radar point cloud")
        self.ax_cloud.set_xlabel("x (m)")
        self.ax_cloud.set_ylabel("y (m)")
        self.ax_cloud.set_zlabel("z (m)")

        if len(points) == 0:
            self.ax_cloud.text2D(0.38, 0.50, "No points", transform=self.ax_cloud.transAxes)
            self.ax_cbar.set_visible(False)
            self._set_cloud_limits()
            return

        self.ax_cbar.set_visible(True)
        color_values = self._get_color_values(points)
        scatter = self.ax_cloud.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color_values,
            cmap="turbo",
            s=12,
            alpha=0.85,
            depthshade=True,
        )
        self._set_cloud_limits()
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(scatter, cax=self.ax_cbar)
        else:
            self.colorbar.update_normal(scatter)
        self.colorbar.set_label(self._colorbar_label())

    def _draw_image(self, image_path: Optional[Path]) -> None:
        """Draws the paired camera image.

        Args:
            image_path: Optional image path.
        """
        self.ax_image.clear()
        self.ax_image.axis("off")
        self.ax_image.set_title("Camera image")

        if image_path is None:
            self.ax_image.text(0.5, 0.5, "No paired image", ha="center", va="center")
            return

        image = self._load_image(str(image_path))
        if image is None:
            self.ax_image.text(0.5, 0.5, "Image load failed", ha="center", va="center")
            return

        self.ax_image.imshow(image)

    def _draw_info(self, frame: FrameItem, points: np.ndarray) -> None:
        """Draws frame-level metadata.

        Args:
            frame: Current frame metadata.
            points: Current point cloud.
        """
        lines = [
            f"Sequence : {self.sequence_dir.name}",
            f"Frame    : {self.frame_idx + 1}/{len(self.frames)}",
            f"Radar    : {frame.bin_path.name}",
            f"Image    : {frame.image_path.name if frame.image_path else 'missing'}",
            f"Points   : {len(points)}",
            f"Color    : {self.color_mode}",
            "Axes     : fixed sequence range",
            f"Status   : {'finished' if self.playback_finished else 'playing' if self.playing else 'paused'}",
            f"Interval : {self.interval_ms} ms",
            "Keys     : Space pause | Left/Right step | Home/End jump",
        ]

        if len(points) > 0:
            xyz_min = points[:, :3].min(axis=0)
            xyz_max = points[:, :3].max(axis=0)
            velocity = points[:, 3]
            snr = points[:, 4]
            lines.extend([
                f"x range  : {xyz_min[0]:7.2f} .. {xyz_max[0]:7.2f} m",
                f"y range  : {xyz_min[1]:7.2f} .. {xyz_max[1]:7.2f} m",
                f"z range  : {xyz_min[2]:7.2f} .. {xyz_max[2]:7.2f} m",
                f"v min/avg/max   : {velocity.min():6.2f} / {velocity.mean():6.2f} / {velocity.max():6.2f} m/s",
                f"snr min/avg/max : {snr.min():6.2f} / {snr.mean():6.2f} / {snr.max():6.2f} dB",
            ])

        self.info_text.set_text("\n".join(lines))

    def _get_color_values(self, points: np.ndarray) -> np.ndarray:
        """Returns the scalar values used for point coloring.

        Args:
            points: Current point cloud.

        Returns:
            One scalar per point.
        """
        if self.color_mode == "snr":
            return points[:, 4]
        if self.color_mode == "z":
            return points[:, 2]
        return points[:, 3]

    def _colorbar_label(self) -> str:
        """Returns a readable label for the current color mode."""
        if self.color_mode == "snr":
            return "snr (dB)"
        if self.color_mode == "z":
            return "z (m)"
        return "velocity (m/s)"

    def _set_default_cloud_limits(self) -> None:
        """Sets a stable view range for empty frames."""
        self.ax_cloud.set_xlim(-8, 8)
        self.ax_cloud.set_ylim(0, 20)
        self.ax_cloud.set_zlim(-1, 5)
        self.ax_cloud.view_init(elev=18, azim=-75)

    def _set_cloud_limits(self) -> None:
        """Applies the fixed sequence-level 3D axis limits."""
        xlim, ylim, zlim = self.cloud_limits
        self.ax_cloud.set_xlim(*xlim)
        self.ax_cloud.set_ylim(*ylim)
        self.ax_cloud.set_zlim(*zlim)
        self.ax_cloud.view_init(elev=18, azim=-75)

    def _sync_slider(self) -> None:
        """Moves the frame slider without recursively redrawing."""
        self._suppress_slider = True
        self.slider.set_val(self.frame_idx)
        self._suppress_slider = False

    def _on_timer(self) -> bool:
        """Advances playback when the timer fires.

        Returns:
            ``True`` to keep the timer callback alive.
        """
        if self.playing:
            if self.frame_idx >= len(self.frames) - 1:
                self._finish_playback()
                return True
            self._draw_frame(self.frame_idx + 1)
        return True

    def _step(self, delta: int) -> None:
        """Moves by a relative number of frames.

        Args:
            delta: Signed frame offset.
        """
        next_idx = int(np.clip(self.frame_idx + delta, 0, len(self.frames) - 1))
        if delta != 0:
            self.playback_finished = False
            self.status_text.set_text("")
        self._draw_frame(next_idx)

    def _toggle_play(self) -> None:
        """Toggles playback."""
        if self.playback_finished:
            self.playback_finished = False
            self.status_text.set_text("")
            self._draw_frame(0)

        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        self.fig.canvas.draw_idle()

    def _finish_playback(self) -> None:
        """Stops playback and shows an end-of-sequence message."""
        self.playing = False
        self.playback_finished = True
        self.timer.stop()
        self.btn_play.label.set_text("Replay")
        self.status_text.set_text("Playback finished.")
        self._draw_frame(len(self.frames) - 1)

    def _change_speed(self, factor: float) -> None:
        """Changes playback speed by scaling the timer interval.

        Args:
            factor: Multiplicative interval scale.
        """
        self.interval_ms = int(np.clip(round(self.interval_ms * factor), 10, 2000))
        self.timer.interval = self.interval_ms
        self._draw_frame(self.frame_idx)

    def _on_slider_changed(self, value: float) -> None:
        """Handles manual slider updates.

        Args:
            value: Slider value.
        """
        if self._suppress_slider:
            return
        self._draw_frame(int(value))

    def _on_color_changed(self, label: str) -> None:
        """Handles point-cloud color mode changes.

        Args:
            label: Selected radio button label.
        """
        self.color_mode = label
        self._draw_frame(self.frame_idx)

    def _on_key_press(self, event) -> None:
        """Handles keyboard shortcuts.

        Args:
            event: Matplotlib key press event.
        """
        if event.key == " ":
            self._toggle_play()
        elif event.key == "left":
            self._step(-1)
        elif event.key == "right":
            self._step(1)
        elif event.key == "home":
            self._draw_frame(0)
        elif event.key == "end":
            self._draw_frame(len(self.frames) - 1)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="View mmWave radar point clouds and paired camera images.", )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help="Acquisition folder that contains image/ and pointcloud/.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Default root used by the folder selector.",
    )
    parser.add_argument(
        "--install-angle",
        type=float,
        default=DEFAULT_ANGLE,
        help="Radar installation pitch angle in degrees.",
    )
    parser.add_argument(
        "--radar-height",
        type=float,
        default=DEFAULT_HEIGHT,
        help="Radar installation height in meters.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=80,
        help="Playback interval in milliseconds.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the radar/image viewer."""
    args = parse_args()
    sequence_dir = args.sequence_dir
    if sequence_dir is None:
        sequence_dir = _find_sequence_dir(args.data_root)
    if sequence_dir is None:
        raise SystemExit("No sequence folder selected. Use --sequence-dir to pass one directly.")

    frames = load_sequence(sequence_dir)
    matched_images = sum(frame.image_path is not None for frame in frames)
    print(f"Loaded {len(frames)} radar frames from: {sequence_dir}")
    print(f"Matched {matched_images} paired images.")

    viewer = MMRadarViewer(
        frames=frames,
        sequence_dir=sequence_dir,
        install_angle=args.install_angle,
        radar_height=args.radar_height,
        interval_ms=args.interval_ms,
    )
    viewer.show()


if __name__ == "__main__":
    main()
