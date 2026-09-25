# Vortex — A Sharded, Lock-Free Event Pipeline in Go, Built Twice

A self-contained, single-file (`main.go`) demonstration of how a "naive" event-processing
pipeline — mutex queue + channel semaphore + buffered file I/O — becomes a **4× faster**
"optimal" pipeline using classic performance-engineering techniques. No external
dependencies; the Go standard library only.

## Run it

```bash
go run vertex/main.go
```

> Run on a machine with **≥ 4 cores** for representative numbers. On a single-core
> sandbox the goroutines time-share one core and every effect is understated.

## What it demonstrates

| # | Technique | Naive | Optimal |
|---|-----------|-------|---------|
| 1 | Cache-line padding | counters sharing one line (~3.5× slower) | each hot field on its own 64-byte line |
| 2 | Semaphore | buffered channel (`ChanSemaphore`) | padded atomic CAS loop (`PaddedSemaphore`) |
| 3 | Sharding | one shared queue | `NumCPU` independent shards, hash-routed |
| 4 | Queue | mutex-guarded slice (`NaiveQueue`) | lock-free SPSC ring buffer (`SPSCRingBuffer`) |
| 5 | Load balancing | none | Chase-Lev work-stealing deque (`WSDeque`) |
| 6 | Persistence | `binary.Write` under a mutex | `mmap` + `unsafe.Slice` zero-copy store |

## Sample output (8-core x86-64)

```
=== False sharing: two goroutines each hammering their own counter ===
unpadded (both counters on one cache line): 629.217768ms
padded   (each counter on its own line):    202.596917ms
padding speedup: 3.11x

generating 4000000 synthetic events across up to 997 shard keys, 8 shards, GOMAXPROCS=8

=== Naive pipeline: mutex queue + channel semaphore + buffered file I/O ===
processed=4000000/4000000 duration=5.680929314s throughput=704110 events/sec

=== Optimal pipeline: sharded SPSC rings + work-stealing deques + padded atomic semaphores + mmap ===
processed=4000000/4000000 duration=1.259856958s throughput=3174964 events/sec steals=206314

end-to-end speedup: 4.51x
```

## Architecture

```
            events in
                │
        ┌───────▼──────────────────────┐
        │  demux (1 goroutine)          │       
        │ shard = hashKey(ShardKey) % N │
        └───┬───┬───┬────────────┬──────┘
            │   │   │            │
      ┌─────▼┐ ┌─▼───┐      ┌────────┐
      │ring 0│ │ring 1│ .... │ring N-1│   SPSC ring buffers (lock-free)
      └──┬───┘ └──┬───┘      └──┬─────┘
         │        │             │ 
      ┌──▼───┐ ┌──▼───┐     ┌──▼────┐
      │deque │ │deque │ ...  │deque  │   Chase-Lev work-stealing deques
      └──┬───┘ └──┬───┘      └──┬────┘
      worker   worker         worker      one per shard; steal when idle
           │
       mmap'd store                        zero-copy, no per-record syscall
```

Flow control: one padded atomic semaphore **per shard** bounds in-flight work
(ring + deque + in-processing) end to end — the single tunable knob.

## Key invariants (why the lock-free parts are safe)

- `SPSCRingBuffer`: exactly **one producer** (demux) and **one consumer** (shard worker)
  per ring → no CAS needed; acquire/release ordering on cursor loads/stores suffices.
- `WSDeque`: `top` only ever moves forward, only via CAS → owner-vs-thief races are
  resolved by exactly one winner.
- `MMapStore`: each record lands in a slot claimed by one atomic increment → workers
  never write the same slot; file size must equal `capacity * sizeof(ResultRecord)`
  (this is the `unsafe` contract).
- Shutdown: workers exit only when `done == 1` **and** global `pending == 0`.

## Caveats (intentional, for pedagogy)

- Demo, not production: no msync/fsync durability guarantee on the mmap path,
  bounded deque drops to inline processing on overflow, and checksum work is
  synthetic. Real services would address durability explicitly.
