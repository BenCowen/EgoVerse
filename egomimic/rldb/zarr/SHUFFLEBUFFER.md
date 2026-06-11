# shufflebuffer loader (`data=mecka_all_sb`)

`SBShardIterableDataset` streams pre-baked per-sample tar shards through the
[`shufflebuffer`](https://github.com/modal-projects/shufflebuffer) drip-fed shuffle buffer. It
yields the **same** sample dict as the zarr/zip path — it runs the same `transform_list` — so the
model, training loop, and norm-stats wiring are unchanged.

## Why

The per-sample cost is a JPEG decode + a small npy load — no zarr open, no chunk decode, no
per-sample random reads — and shards are read sequentially. So the loader keeps the GPU fed (the
training stays compute-bound) without staging whole episodes to a large NVMe pool. The buffer
delivers a globally well-mixed stream: global uniformity comes from random sample→shard assignment
at build time, the buffer kills the remaining local correlation.

## Install

```
pip install -r requirements.txt   # pulls shufflebuffer (pinned commit) from GitHub
```

## Run

```
python -m egomimic.trainHydra model=hpt_bc_flow_mecka data=mecka_all_sb trainer=ddp \
    norm_stats.precomputed_norm_path=precomputed_norm_stats/mecka_all_zarr \
    data.train_datasets.mecka_bimanual.resolver.shard_dir=/path/to/shards
```

Multi-GPU is automatic: shards are partitioned across `world_size × num_workers` consumers
(disjoint, reshuffled per epoch identically on every rank). Requirement:
`#shards ≥ world_size × num_workers`. Equal step counts per rank come from the trainer's
`limit_train_batches` (as on the zip/zarr path); alternatively set
`samples_per_epoch` on the dataset and shufflebuffer enforces it.

## Shard format

WebDataset-style tars of per-sample records: `<key>.jpg` (the image) + `<key>.npy` (a pickled dict
of the raw key_map inputs — `action_l`, `action_r`, `proprio_l`, `proprio_r`, `proprio_head`).
`decode()` rebuilds the raw dict and runs `transform_list`, producing the identical sample.

**Precondition for good mixing:** samples must be assigned to shards **at random** when the shards
are built (interleaved across episodes), not one episode per shard. With content-contiguous shards
the buffer's mixing ceiling is `buffer_size / dataset_size`; `shufflebuffer.warn_if_contiguous`
flags that case. Producing the shards (the materializer) is upstream and separate from this loader.

## Knobs

- `buffer_size` — reservoir size in samples; the memory cap (≈ `buffer_size × ~62 KB`) and the
  effective shuffle window. Since shards are already randomly assigned, a modest buffer
  (10–20k) is globally representative.
- `prefetch` — whole shards staged ahead (`2` = double-buffer).
- `num_workers` / `prefetch_factor` — standard DataLoader knobs; decode parallelizes across workers.
