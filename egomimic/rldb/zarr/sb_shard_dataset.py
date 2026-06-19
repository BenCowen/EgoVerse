"""Feed the model from pre-baked per-sample tar shards via the shufflebuffer reservoir loader.

A drop-in alternative to the zarr/zip path: instead of opening an episode and reading frames, it
streams pre-materialized WebDataset-style records (``<key>.jpg`` + ``<key>.npy``, one self-contained
sample each) through shufflebuffer's drip-fed shuffle buffer. ``decode`` rebuilds the raw key_map
inputs and runs the SAME ``transform_list``, so the sample dict is identical to ZarrDataset's
(``observations.images.front_img_1`` (3,360,640) f32, ``actions_cartesian`` (100,12) f32,
``observations.state.ee_pose`` (12,) f32, ``embodiment`` / ``metadata.robot_name`` ints) — the
model, training loop, and norm-stats wiring are unchanged.

Per-sample cost is a JPEG decode + a small npy load (no zarr open, no chunk decode), and shards are
read sequentially, so the loader keeps the GPU fed without the per-sample random reads the zarr
path pays. Shards are built once with random sample->shard assignment (see SHUFFLEBUFFER.md), which
is what lets the buffer deliver a globally well-mixed stream.

Selectable as ``data=mecka_all_sb``. The shard format is produced upstream by the materializer and
is independent of this loader.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
from shufflebuffer import (
    ShardPool,
    list_shards,
    partition_shards,
    resolve_dist,
    take_exactly,
    tar_load_shard,
)

from egomimic.rldb.embodiment.embodiment import get_embodiment_id

VIZ_IMAGE_KEY = "observations.images.front_img_1"


class ShardResolver:
    """Carries the shard location + key_map / transform_list / norm_stats, mirroring
    ZipEpisodeResolver so the trainHydra norm-stats path (which deep-copies the config and sets
    ``resolver.key_map["norm_mode"] = True``) works against this loader unchanged.

    The norm-stats variant of the key_map drops the camera/annotation keys, so an instantiated
    key_map without the image key signals the norm pass — see ``is_norm_mode``."""

    def __init__(
        self,
        shard_dir: str,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
        valid_ratio: float = 0.1,
        seed: int = 42,
        debug: int | None = None,
    ):
        self.shard_dir = Path(shard_dir)
        self.key_map = key_map
        self.transform_list = transform_list or []
        self.norm_stats = norm_stats
        self.pause_removal_epsilon = pause_removal_epsilon
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.debug = debug
        self._all = list_shards(str(self.shard_dir))
        if not self._all:
            raise RuntimeError(f"No *.tar shards found in {self.shard_dir}")

    def split(self, mode: str) -> list[str]:
        rng = random.Random(self.seed)
        shards = list(self._all)
        rng.shuffle(shards)
        if self.debug:
            shards = shards[: max(1, int(self.debug))]
        n_valid = max(1, int(len(shards) * self.valid_ratio))
        if mode == "valid":
            return shards[:n_valid]
        if mode == "train":
            return shards[n_valid:]
        return shards

    def is_norm_mode(self) -> bool:
        """True when key_map was built with norm_mode=True (camera keys dropped) → plain reader."""
        km = self.key_map
        if not km:
            return False
        try:
            keys = set(km.keys())
        except Exception:
            return False
        return len(keys) > 0 and VIZ_IMAGE_KEY not in keys


class SBShardIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        resolver: ShardResolver,
        mode: str = "train",
        buffer_size: int = 20000,
        prefetch: int = 2,
        prefetch_workers: int = 4,
        seed: int = 42,
        embodiment_name: str = "mecka_bimanual",
        samples_per_epoch: int | None = None,
    ):
        super().__init__()
        self.resolver = resolver
        self.key_map = resolver.key_map
        self.transform_list = resolver.transform_list
        self.norm_stats = resolver.norm_stats
        # The norm-stats pre-pass only needs to SEE samples, not shuffle them, so it gets a plain
        # sequential reader (no reservoir warmup, no continuous prefetch); training uses the full
        # reservoir. Decoupled entirely so neither regresses the other.
        self.norm_mode = resolver.is_norm_mode()
        self.mode = mode
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.prefetch_workers = prefetch_workers
        self.seed = seed
        # Optional DDP step budget (total across ranks). When set, shufflebuffer holds every rank
        # to an identical count so all-reduce can't hang; leave None to rely on the trainer's
        # limit_train_batches (as the zip/zarr path does).
        self.samples_per_epoch = samples_per_epoch
        self.embodiment_id = get_embodiment_id(embodiment_name)
        self._shards = resolver.split(mode)
        self._epoch = 0
        self.data_schematic = None

    # -- trainHydra / DataSchematic wiring (mirror the zarr/zip datasets) ---------
    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        self.data_schematic = data_schematic
        if getattr(data_schematic, "norm_stats", None) is not None and self.norm_stats is None:
            self.norm_stats = data_schematic.norm_stats

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        # Approximate (Lightning progress only); ~16k samples per ~1 GiB shard.
        return len(self._shards) * 16000

    def __getitem__(self, idx: int) -> dict:
        # trainHydra calls dataset[0] once for shape inference. Decode one sample from the first
        # shard — no reservoir warmup. (Norm-stats uses the DataLoader/__iter__ path, not this.)
        return self._decode(tar_load_shard(self._shards[0])[0])

    # -- per-record decode: rebuild raw key_map inputs, run the real transform_list --
    def _decode(self, rec: dict) -> dict:
        meta = np.load(io.BytesIO(rec["npy"]), allow_pickle=True).item()
        data = {
            "left.action_ee_pose": torch.as_tensor(meta["action_l"], dtype=torch.float32),
            "right.action_ee_pose": torch.as_tensor(meta["action_r"], dtype=torch.float32),
            "left.obs_ee_pose": torch.as_tensor(meta["proprio_l"], dtype=torch.float32),
            "right.obs_ee_pose": torch.as_tensor(meta["proprio_r"], dtype=torch.float32),
            "obs_head_pose": torch.as_tensor(meta["proprio_head"], dtype=torch.float32),
            "embodiment": self.embodiment_id,
            "metadata.robot_name": self.embodiment_id,
        }
        if not self.norm_mode:                       # norm-stats needs only pose/action keys
            import simplejpeg
            img = simplejpeg.decode_jpeg(rec["jpg"])                   # HWC uint8
            data[VIZ_IMAGE_KEY] = torch.from_numpy(
                np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0   # (3,360,640) f32
        for t in self.transform_list:                                  # → actions_cartesian etc.
            data = t.transform(data)
        # Match ZarrDataset: emit float32 tensors. The pose transforms run in numpy/float64, so
        # cast their outputs (the int embodiment/metadata scalars are left untouched). Without this
        # the model receives float64 actions — a dtype mismatch against the f32 image + weights.
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32)
            elif isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                data[k] = v.to(torch.float32)
        return data

    def _refs_for_consumer(self, epoch: int) -> list[str]:
        rank, world_size = resolve_dist()
        wi = torch.utils.data.get_worker_info()
        worker_id, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        return partition_shards(
            self._shards, rank=rank, world_size=world_size,
            worker_id=worker_id, num_workers=num_workers, epoch=epoch, seed=self.seed,
        )

    def _plain_iter(self):
        # Norm-stats pass: stream records shard-by-shard, no reservoir/warmup/prefetch — just enough
        # for the stats sampler to see a representative slice (shuffle is irrelevant to mean/std).
        for ref in self._refs_for_consumer(self._epoch):
            try:
                recs = tar_load_shard(ref)
            except Exception:
                continue
            for rec in recs:
                try:
                    yield self._decode(rec)
                except Exception:
                    continue

    def __iter__(self):
        if self.norm_mode:
            yield from self._plain_iter()
            return

        rank, world_size = resolve_dist()
        wi = torch.utils.data.get_worker_info()
        worker_id, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        gid = rank * num_workers + worker_id

        def make_pass(wrap):
            epoch = self._epoch + wrap * 1_000_003
            refs = partition_shards(
                self._shards, rank=rank, world_size=world_size,
                worker_id=worker_id, num_workers=num_workers, epoch=epoch, seed=self.seed,
            )
            return ShardPool(
                refs, tar_load_shard, buffer_size=self.buffer_size, prefetch=self.prefetch,
                prefetch_workers=self.prefetch_workers,
                seed=(hash((self.seed, epoch, gid)) & 0xFFFFFFFF),
            )

        # A single corrupt record shouldn't kill the worker — skip it (never yield None, which the
        # default collate can't batch).
        recs = make_pass(0) if self.samples_per_epoch is None else take_exactly(
            make_pass, max(1, self.samples_per_epoch // (world_size * num_workers))
        )
        for rec in recs:
            try:
                yield self._decode(rec)
            except Exception:
                continue
