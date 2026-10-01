# Apache Druid vs ClickHouse – single-node benchmark (4 cores)

## Layout
```
docker-compose.yml     Druid (all services) + ClickHouse, selected via compose profiles
.env                   ALL knobs: cpuset, memory, versions, dataset scale, loader parallelism
druid/environment      Druid runtime config
clickhouse/schema.sql  ClickHouse table (sort key / partitioning – edit to match your prod table)
queries/queries.json   Benchmark queries (shared SQL, per-engine override where dialects differ)
scripts/generate_data.py   Synthetic data -> data/events_YYYY-MM-DD.csv (+ meta.json)
scripts/load_druid.py      Native batch ingestion, waits until segments are queryable
scripts/load_clickhouse.sh Schema + parallel load (+ OPTIMIZE FINAL)
scripts/benchmark.py       Latency percentiles x time-range scale x concurrency
scripts/compare.py         Side-by-side Druid vs ClickHouse table
scripts/run_matrix.sh      Fully automated run across several dataset sizes
```

## CPU bounding
Every container of an engine gets `cpuset: 0-3` (`CPUSET` in `.env`). Unlike `cpus: 4`, this caps the
*whole* Druid stack (broker, historical, middleManager, ZK, Postgres...) to the same 4 cores instead of
4 cores each. JVMs and ClickHouse both see 4 CPUs. Run **one engine at a time** so each gets all 4 cores.

## Quick start
```bash
pip install -r requirements.txt

# 1) generate data (scale via .env or inline)
ROWS=10000000 DAYS=30 python3 scripts/generate_data.py

# 2) ClickHouse
make up-clickhouse && make load-clickhouse && make bench-clickhouse && make down

# 3) Druid
make up-druid && make load-druid && make bench-druid && make down

# 4) compare
make compare
```
Or everything, across scales: `SCALES="10000000 50000000 100000000" bash scripts/run_matrix.sh --iterations 200`

## Scales
* **Data scale**: `ROWS`, `DAYS`, `USERS` (user_id cardinality), `CAMPAIGNS`, `PAGES` in `.env`.
  Data is skewed (Zipf-like) so group-bys and filters behave realistically. `make reset` wipes volumes.
* **Query scale**: `--scales 6h,1d,7d,all` = how much of the time range each query scans. Each iteration
  uses a different random window (identical windows for both engines, seeded) to avoid trivially
  repeating the same query.
* **Concurrency**: `--concurrency 1,4,16,32`.
* **Percentiles**: `--percentiles 50,90,95,99,99.9`. Use `--iterations` ≥ 100 for p99, ≥ 1000 for p99.9.

Examples:
```bash
python3 scripts/benchmark.py --engines druid --queries 'topn|groupby' --scales 1d,all --concurrency 1,16 --iterations 500
python3 scripts/benchmark.py --engines clickhouse --percentiles 50,95,99,99.9 --iterations 1000
```
Results go to `results/<engine>_<rows>_<timestamp>.csv` (ms).

## Fairness notes – read before drawing conclusions
* Druid ingests with **rollup=false** so both engines store the same raw rows. If your prod jobs roll up,
  enable rollup in `load_druid.py` – Druid will usually look much better (fewer rows).
* ClickHouse's `ORDER BY` in `schema.sql` is the main tuning lever; Druid's segment layout is day-by-day,
  time-sorted. Try the sort key your real queries want.
* `COUNT(DISTINCT)` → Druid is approximate (HLL); ClickHouse uses `uniq` (approximate) to match.
  Druid TopN/`GROUP BY … ORDER BY … LIMIT` may be approximate (`useApproximateTopN`).
* Both engines rely on the OS page cache. First runs of a scale are "warm-ish" after loading; for cold-cache
  numbers restart the container and drop caches (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`).
* Client-side latency includes HTTP + JSON parsing; same for both. Druid's broker adds a network hop by design.
* Memory: Druid heap+direct ≈ 7 GB in total (see compose), ClickHouse is capped by `CH_MEM_LIMIT`. Keep them comparable.
* Druid version `34.0.0`: bump `DRUID_VERSION` if the tag isn't available or you want to match prod.
* Also test **ingestion** time: both loaders print elapsed time.
* On Docker Desktop (Mac/Windows), `cpuset` applies inside the Linux VM; give the VM ≥ 4 CPUs. Prefer a Linux host for real numbers.
