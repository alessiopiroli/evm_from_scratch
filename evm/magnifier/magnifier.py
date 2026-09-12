from evm.utils.misc import create_magnified_video, get_pyramids, isolate_frequencies, load_video, magnificate


class Magnifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_levels = self.cfg.MAGNIFICATION.n_levels
        self.lowcut = float(self.cfg.MAGNIFICATION.lowcut)
        self.range = float(self.cfg.MAGNIFICATION.range)
        self.alpha = float(self.cfg.MAGNIFICATION.alpha)

    def magnify(self, video_path, output_path):
        frames, fps = load_video(video_path)
        pyramids = get_pyramids(frames, self.n_levels)
        filtered = isolate_frequencies(pyramids, self.lowcut, self.range, fps)
        amplified_pyramid = magnificate(pyramids, filtered, self.alpha)
        create_magnified_video(amplified_pyramid, output_path, fps, frames)
