"""Data pipeline: window sampling, the dataset, collate and normalisation statistics.

    dataloader.py, sampling.py   the dataset, sampler and collate; how windows are drawn
    stats.py                     normalisation statistics and their consistency checks
    robot_core.py, vitra_core/   episode readers: robot joints; hand poses (from VITRA)
"""
from .dataloader import (SequenceDataset, Source, MixtureSampler, JpegFrameStore,
                         sources_from_config,
                         Unit, make_collate,
                         to_observations, load_video_list)
from .sampling import Window, sample_window, length_weights, supervisable_anchors, phase_anchors
from .stats import (load_statistics, load_v4_statistics, check_convention,
                    check_sources, split_digest, report, SCHEMA_VERSION)

__all__ = ["SequenceDataset", "Source", "MixtureSampler", "JpegFrameStore",
           "sources_from_config", "Unit", "make_collate",
           "to_observations", "load_video_list",
           "Window", "sample_window", "length_weights", "supervisable_anchors",
           "phase_anchors",
           "load_statistics", "load_v4_statistics", "check_convention",
           "check_sources", "split_digest", "report", "SCHEMA_VERSION"]
