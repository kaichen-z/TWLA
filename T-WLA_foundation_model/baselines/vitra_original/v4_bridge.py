"""The loader of this baseline, and the path resolution of the run json.

`dataloader/` in this package is a copy of `T-WLA_foundation_model/dataloader/` fixed to the
joint-angle action layout. `_ensure_path` checks that `import dataloader` resolved to
that copy, because a loader with the other layout would not raise.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _ensure_path() -> None:
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    import dataloader as _dl

    got = os.path.dirname(os.path.abspath(_dl.__file__))
    want = os.path.join(HERE, "dataloader")
    if got != want:
        raise RuntimeError(
            f"`import dataloader` resolved to {got}, not this baseline's copy at {want}. The two differ in the "
            f"action layout, and the wrong one does not raise. Check sys.path order.")


_ensure_path()

from dataloader import (MixtureSampler, SequenceDataset, check_convention,  # noqa: E402
                        check_sources, load_v4_statistics, load_video_list,
                        make_collate, sources_from_config, to_observations)


def resolve_paths(C: dict) -> dict:
    """Expand `${VAR}` and `~` in the paths of a run json and root the relative ones.

    The entries of `data`, including the paths inside `data.sources`, are relative to
    `data.root` (normally `${VLA_DATA}`) unless they are absolute. `output_dir` is
    expanded but not rooted.
    """
    def expand(v):
        return os.path.expanduser(os.path.expandvars(v)) if isinstance(v, str) else v

    C = dict(C)
    C["output_dir"] = expand(C["output_dir"])
    D = dict(C.get("data", {}))
    root = expand(D.pop("root", ""))
    if "${" in root or "${" in C["output_dir"]:
        raise KeyError(
            f"unset environment variable in a path: root={root!r}, "
            f"output_dir={C['output_dir']!r}. Set it (e.g. `export VLA_DATA=...`) or "
            f"write an absolute path into the config.")
    def rooted(v):
        if v is None:
            return None
        v = expand(v)
        return v if os.path.isabs(v) else os.path.join(root, v)

    # The path entries of a source; the others (`name`, `weight`) are left alone.
    SOURCE_PATHS = {"video_root", "annotation_file", "label_folder", "jpeg_root",
                    "split_dir", "statistics_split_dir"}
    srcs = D.pop("sources", None)
    for k, v in D.items():
        D[k] = rooted(v)
    if srcs is not None:
        D["sources"] = [{k: (rooted(v) if k in SOURCE_PATHS else v) for k, v in s.items()}
                        for s in srcs]
    C["data"] = D
    return C


__all__ = ["MixtureSampler", "SequenceDataset", "check_convention",
           "check_sources", "load_v4_statistics", "load_video_list", "make_collate",
           "resolve_paths", "sources_from_config", "to_observations"]
