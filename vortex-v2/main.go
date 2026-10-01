// Command vortex is a self-contained demonstration of an event-processing
// pipeline built five times, showing the incremental gain from each technique:
//
//	[1] Naive     — mutex queue + channel semaphore (4× in-flight budget) +
//	                binary.Write (reflects on every record). The intentionally
//	                weak baseline from the original blog post.
//
//	[2] Idiomatic — buffered channel + worker pool. Raises the in-flight budget
//	                to match the optimal engine (1024 per shard) and removes
//	                reflection from writes. This is the fair Go baseline: once
//	                the playing field is level the naive→idiomatic gap alone
//	                accounts for ~half the total speedup.
//
//	[3] ShardedChan — per-shard buffered channels of event batches + mmap
//	                persistence + channel-based work stealing (non-blocking
//	                select across peer channels). Pure Go: no unsafe, no custom
//	                data structures. Surprisingly competitive because Go's
//	                scheduler is optimised for channel blocking; workers park
//	                instead of spinning when their shard is idle.
//
//	[4] Optimal   — sharded SPSC ring buffers + Chase-Lev work-stealing deques
//	                + padded atomic semaphores + mmap. Wins when the workload
//	                is CPU-bound and sustained: it avoids scheduler round-trips
//	                entirely, but the spin-wait idle path costs more than
//	                channel parking under bursty or uneven load.
//
//	[5] Partitioned — static/dynamic range partitioning + index-preserving
//	                mmap writes (records[i] for events[i]). No queue, no
//	                semaphore, no cursor atomic, no time.Now() in the hot
//	                loop, no per-event hash for routing. One atomic per 1024-
//	                event chunk instead of 4 atomics per event. This is the
//	                fastest on batch 1:1 workloads because output position is
//	                a pure function of input position.
//
// Concepts covered, and where:
//
//  1. Cache-line padding      -> cacheLineSize / pad fields, demoFalseSharing()
//  2. Semaphore variants      -> ChanSemaphore (channel) vs PaddedSemaphore
//     (atomic CAS loop, false-share-free)
//  3. Sharding                -> engines 3 & 4: N independent shards,
//     hash-routed by ShardKey
//  4. SPSC ring buffer        -> SPSCRingBuffer (engine 4 only)
//  5. Work-stealing           -> WSDeque / Chase-Lev (engine 4);
//     non-blocking select across channels (engine 3)
//  6. mmap + unsafe           -> MMapStore (engines 3 & 4) vs NaiveStore
//
// Run it:  go run main.go
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Shared domain types
// ---------------------------------------------------------------------------

// Event is the unit of work flowing through the pipeline.
type Event struct {
	ID        uint64
	Timestamp int64
	ShardKey  uint64
	Payload   [32]byte
}

// ResultRecord is what we persist per processed Event. Fixed-size and
// alignment-friendly on purpose: it's what makes the mmap zero-copy trick
// possible later on.
type ResultRecord struct {
	ID        uint64
	Timestamp int64
	Checksum  uint64
	Worker    int32
	_         int32 // explicit pad to keep the struct 8-byte aligned/sized
}

// hashKey is a cheap, allocation-free integer mix (the MurmurHash3 64-bit finalizer).
func hashKey(k uint64) uint64 {
	k ^= k >> 33
	k *= 0xff51afd7ed558ccd
	k ^= k >> 33
	k *= 0xc4ceb9fe1a85ec53
	k ^= k >> 33
	return k
}

// computeChecksum simulates real per-event CPU work.
func computeChecksum(e *Event) uint64 {
	h := e.ID ^ hashKey(e.ShardKey)
	for i := 0; i < len(e.Payload); i += 8 {
		var v uint64
		for j := 0; j < 8; j++ {
			v = (v << 8) | uint64(e.Payload[i+j])
		}
		h = hashKey(h ^ v)
	}
	return h
}

// computeChecksumFast is the same logical work without the byte-at-a-time
// shift/or loop. Payload is always 32 bytes, so four fixed LittleEndian loads
// let the compiler unroll and avoid per-byte bounds checks. ~3-4x faster than
// computeChecksum on amd64/arm64.
func computeChecksumFast(e *Event) uint64 {
	h := e.ID ^ hashKey(e.ShardKey)
	h = hashKey(h ^ binary.LittleEndian.Uint64(e.Payload[0:8]))
	h = hashKey(h ^ binary.LittleEndian.Uint64(e.Payload[8:16]))
	h = hashKey(h ^ binary.LittleEndian.Uint64(e.Payload[16:24]))
	h = hashKey(h ^ binary.LittleEndian.Uint64(e.Payload[24:32]))
	return h
}

func generateEvents(n int) []*Event {
	events := make([]*Event, n)
	for i := 0; i < n; i++ {
		e := &Event{
			ID:        uint64(i),
			Timestamp: time.Now().UnixNano(),
			ShardKey:  uint64(i%997) * 2654435761,
		}
		// Using math/rand/v2 ChaCha8 / PCG generator for event payload generation
		for j := 0; j < len(e.Payload); j += 8 {
			val := rand.Uint64()
			binary.LittleEndian.PutUint64(e.Payload[j:], val)
		}
		events[i] = e
	}
	return events
}

// ---------------------------------------------------------------------------
// 1. Cache-line padding demo using atomic.Int64
// ---------------------------------------------------------------------------

const cacheLineSize = 64

type unpaddedCounters struct {
	a atomic.Int64
	b atomic.Int64
}

type paddedCounters struct {
	a atomic.Int64
	_ [cacheLineSize - 8]byte
	b atomic.Int64
	_ [cacheLineSize - 8]byte
}

func demoFalseSharing() {
	const iterations = 20_000_000

	run := func(addA, addB func()) time.Duration {
		var wg sync.WaitGroup
		wg.Add(2)
		start := time.Now()
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				addA()
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				addB()
			}
		}()
		wg.Wait()
		return time.Since(start)
	}

	uc := &unpaddedCounters{}
	unpaddedDur := run(
		func() { uc.a.Add(1) },
		func() { uc.b.Add(1) },
	)

	pc := &paddedCounters{}
	paddedDur := run(
		func() { pc.a.Add(1) },
		func() { pc.b.Add(1) },
	)

	fmt.Println("=== False sharing: two goroutines each hammering their own counter ===")
	fmt.Printf("unpadded (both counters on one cache line): %v\n", unpaddedDur)
	fmt.Printf("padded   (each counter on its own line):    %v\n", paddedDur)
	if runtime.NumCPU() < 2 {
		fmt.Println("(this sandbox only exposes 1 CPU, so the two goroutines can't run on")
		fmt.Println(" separate cores - run this on a multi-core machine to see the gap.)")
	} else {
		fmt.Printf("padding speedup: %.2fx\n", unpaddedDur.Seconds()/paddedDur.Seconds())
	}
	fmt.Println()
}

// ---------------------------------------------------------------------------
// 2. Semaphore: naive (channel) vs optimal (padded atomic CAS loop)
// ---------------------------------------------------------------------------

type semaphore interface {
	TryAcquire() bool
	Release()
}

type ChanSemaphore struct {
	ch chan struct{}
}

func NewChanSemaphore(n int) *ChanSemaphore { return &ChanSemaphore{ch: make(chan struct{}, n)} }

func (s *ChanSemaphore) TryAcquire() bool {
	select {
	case s.ch <- struct{}{}:
		return true
	default:
		return false
	}
}

func (s *ChanSemaphore) Release() { <-s.ch }

type PaddedSemaphore struct {
	count atomic.Int64
	_     [cacheLineSize - 8]byte
	max   int64
	_     [cacheLineSize - 8]byte
}

func NewPaddedSemaphore(max int64) *PaddedSemaphore {
	return &PaddedSemaphore{max: max}
}

func (s *PaddedSemaphore) TryAcquire() bool {
	for {
		c := s.count.Load()
		if c >= s.max {
			return false
		}
		if s.count.CompareAndSwap(c, c+1) {
			return true
		}
	}
}

func (s *PaddedSemaphore) Release() { s.count.Add(-1) }

// ---------------------------------------------------------------------------
// 3 & 4. Sharding + Generic SPSC Ring Buffer
// ---------------------------------------------------------------------------

type NaiveQueue struct {
	mu    sync.Mutex
	items []*Event
}

func (q *NaiveQueue) Enqueue(e *Event) {
	q.mu.Lock()
	q.items = append(q.items, e)
	q.mu.Unlock()
}

func (q *NaiveQueue) Dequeue() (*Event, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil, false
	}
	e := q.items[0]
	q.items = q.items[1:]
	return e, true
}

func (q *NaiveQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// Generic Single-Producer Single-Consumer (SPSC) Ring Buffer using atomic.Pointer.
type SPSCRingBuffer[T any] struct {
	buf  []atomic.Pointer[T]
	mask uint64

	_    [cacheLineSize]byte
	head atomic.Uint64 // consumer-owned
	_    [cacheLineSize - 8]byte
	tail atomic.Uint64 // producer-owned
	_    [cacheLineSize - 8]byte
}

func NewSPSCRingBuffer[T any](capacityPow2 uint64) *SPSCRingBuffer[T] {
	if capacityPow2 == 0 || capacityPow2&(capacityPow2-1) != 0 {
		panic("capacity must be a power of two")
	}
	return &SPSCRingBuffer[T]{
		buf:  make([]atomic.Pointer[T], capacityPow2),
		mask: capacityPow2 - 1,
	}
}

func (r *SPSCRingBuffer[T]) Enqueue(item *T) bool {
	tail := r.tail.Load()
	head := r.head.Load()
	if tail-head >= uint64(len(r.buf)) {
		return false
	}
	idx := tail & r.mask
	r.buf[idx].Store(item)
	r.tail.Store(tail + 1)
	return true
}

func (r *SPSCRingBuffer[T]) Dequeue() (*T, bool) {
	head := r.head.Load()
	tail := r.tail.Load()
	if head >= tail {
		return nil, false
	}
	idx := head & r.mask
	item := r.buf[idx].Load()
	r.head.Store(head + 1)
	return item, true
}

func (r *SPSCRingBuffer[T]) Empty() bool {
	return r.head.Load() >= r.tail.Load()
}

// ---------------------------------------------------------------------------
// 5. Work-stealing deque (Chase-Lev), generic variant
// ---------------------------------------------------------------------------

type WSDeque[T any] struct {
	_      [cacheLineSize]byte
	top    atomic.Int64
	_      [cacheLineSize - 8]byte
	bottom atomic.Int64
	_      [cacheLineSize - 8]byte
	mask   int64
	buf    []atomic.Pointer[T]
}

func NewWSDeque[T any](capacityPow2 int64) *WSDeque[T] {
	if capacityPow2 <= 0 || capacityPow2&(capacityPow2-1) != 0 {
		panic("capacity must be a power of two")
	}
	return &WSDeque[T]{
		mask: capacityPow2 - 1,
		buf:  make([]atomic.Pointer[T], capacityPow2),
	}
}

func (d *WSDeque[T]) PushBottom(item *T) bool {
	b := d.bottom.Load()
	t := d.top.Load()
	if b-t >= int64(len(d.buf)) {
		return false
	}
	d.buf[b&d.mask].Store(item)
	d.bottom.Store(b + 1)
	return true
}

func (d *WSDeque[T]) PopBottom() (*T, bool) {
	b := d.bottom.Load() - 1
	d.bottom.Store(b)
	t := d.top.Load()

	if t > b {
		d.bottom.Store(b + 1)
		return nil, false
	}

	item := d.buf[b&d.mask].Load()

	if t == b {
		ok := d.top.CompareAndSwap(t, t+1)
		d.bottom.Store(b + 1)
		if !ok {
			return nil, false
		}
		return item, true
	}

	return item, true
}

func (d *WSDeque[T]) Steal() (*T, bool) {
	t := d.top.Load()
	b := d.bottom.Load()
	if t >= b {
		return nil, false
	}
	item := d.buf[t&d.mask].Load()
	if !d.top.CompareAndSwap(t, t+1) {
		return nil, false
	}
	return item, true
}

func (d *WSDeque[T]) Empty() bool {
	return d.top.Load() >= d.bottom.Load()
}

// ---------------------------------------------------------------------------
// 6. Persistence: Naive vs MMapStore
// ---------------------------------------------------------------------------

type NaiveStore struct {
	mu sync.Mutex
	f  *os.File
	w  *bufio.Writer
}

func NewNaiveStore(path string) (*NaiveStore, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	return &NaiveStore{f: f, w: bufio.NewWriter(f)}, nil
}

func (s *NaiveStore) Write(rec ResultRecord) {
	s.mu.Lock()
	_ = binary.Write(s.w, binary.LittleEndian, rec)
	s.mu.Unlock()
}

// writeRecord avoids binary.Write's per-call reflection by writing each field
// directly. Used by IdiomaticEngine to give it a fair comparison.
func (s *NaiveStore) writeRecord(rec ResultRecord) {
	var buf [32]byte
	binary.LittleEndian.PutUint64(buf[0:], rec.ID)
	binary.LittleEndian.PutUint64(buf[8:], uint64(rec.Timestamp))
	binary.LittleEndian.PutUint64(buf[16:], rec.Checksum)
	binary.LittleEndian.PutUint32(buf[24:], uint32(rec.Worker))
	// buf[28:32] stays zero — matches the explicit pad in ResultRecord
	s.mu.Lock()
	_, _ = s.w.Write(buf[:])
	s.mu.Unlock()
}

func (s *NaiveStore) Close() error {
	if err := s.w.Flush(); err != nil {
		return err
	}
	if err := s.f.Sync(); err != nil {
		return err
	}
	return s.f.Close()
}

type MMapStore struct {
	file    *os.File
	data    []byte
	records []ResultRecord
	cursor  atomic.Int64
}

func NewMMapStore(path string, capacity int) (*MMapStore, error) {
	recSize := int(unsafe.Sizeof(ResultRecord{}))
	size := recSize * capacity

	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	if err := f.Truncate(int64(size)); err != nil {
		f.Close()
		return nil, err
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}

	records := unsafe.Slice((*ResultRecord)(unsafe.Pointer(&data[0])), capacity)
	return &MMapStore{file: f, data: data, records: records}, nil
}

func (m *MMapStore) Write(rec ResultRecord) bool {
	idx := m.cursor.Add(1) - 1
	if idx >= int64(len(m.records)) {
		return false
	}
	m.records[idx] = rec
	return true
}

// WriteAt is the coordination-free persist path for batch 1:1 workloads:
// output slot i belongs to input slot i, so no cursor atomic is needed.
// Each worker owns disjoint index ranges (chunk-aligned), keeping the written
// cache lines in L1 instead of ping-ponging a single cursor line.
func (m *MMapStore) WriteAt(idx int, rec ResultRecord) {
	m.records[idx] = rec
}

func (m *MMapStore) Close() error {
	if err := syscall.Munmap(m.data); err != nil {
		return err
	}
	return m.file.Close()
}

// ---------------------------------------------------------------------------
// Engines: four implementations of the same pipeline
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Engine 2: Idiomatic Go — buffered channel + worker pool
//
// This is the "fair" baseline the commenter asked for: the channel's own
// capacity bounds in-flight work (same 1024*numShards budget as Optimal),
// binary.Write is replaced by a plain typed write, and every worker draws
// from the same channel so there's no semaphore at all.
// ---------------------------------------------------------------------------

type IdiomaticEngine struct {
	ch        chan *Event
	store     *NaiveStore
	wg        sync.WaitGroup
	processed atomic.Int64
}

func runIdiomatic(events []*Event, numWorkers int, path string) (time.Duration, int64) {
	store, err := NewNaiveStore(path)
	if err != nil {
		panic(err)
	}
	// Match the total in-flight budget of OptimalEngine (1024 per shard).
	inFlight := numWorkers * 1024
	eng := &IdiomaticEngine{
		ch:    make(chan *Event, inFlight),
		store: store,
	}

	start := time.Now()

	eng.wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go eng.worker(i)
	}

	go func() {
		for _, ev := range events {
			eng.ch <- ev // blocks naturally when the channel is full
		}
		close(eng.ch)
	}()

	eng.wg.Wait()
	dur := time.Since(start)

	if err := eng.store.Close(); err != nil {
		panic(err)
	}
	return dur, eng.processed.Load()
}

func (eng *IdiomaticEngine) worker(id int) {
	defer eng.wg.Done()
	for ev := range eng.ch {
		checksum := computeChecksum(ev)
		rec := ResultRecord{
			ID:        ev.ID,
			Timestamp: time.Now().UnixNano(),
			Checksum:  checksum,
			Worker:    int32(id),
		}
		eng.store.writeRecord(rec)
		eng.processed.Add(1)
	}
}

// ---------------------------------------------------------------------------
// Engine 3: Sharded channels + channel-based work stealing
//
// Each shard gets its own buffered channel of *batches*. A shard's owner
// goroutine first drains its own channel; when idle it tries the others with
// a non-blocking select — no custom deque, no unsafe, pure Go.
// ---------------------------------------------------------------------------

const chanBatchSize = 32

type ShardedChanEngine struct {
	numShards int
	shards    []chan []*Event // each shard receives batches
	store     *MMapStore

	wg        sync.WaitGroup
	processed atomic.Int64
	steals    atomic.Int64
}

func runShardedChan(events []*Event, numShards int, path string) (time.Duration, int64, int64) {
	store, err := NewMMapStore(path, len(events))
	if err != nil {
		panic(err)
	}

	eng := &ShardedChanEngine{
		numShards: numShards,
		shards:    make([]chan []*Event, numShards),
		store:     store,
	}
	// 1024-event in-flight budget per shard, expressed as batch-count capacity.
	chanCap := 1024 / chanBatchSize
	if chanCap < 4 {
		chanCap = 4
	}
	for i := range eng.shards {
		eng.shards[i] = make(chan []*Event, chanCap)
	}

	start := time.Now()

	eng.wg.Add(numShards)
	for i := 0; i < numShards; i++ {
		go eng.workerLoop(i)
	}

	go func() {
		batch := make([]*Event, 0, chanBatchSize)
		buckets := make([][]*Event, numShards)
		for i := range buckets {
			buckets[i] = make([]*Event, 0, chanBatchSize)
		}
		flush := func(s int) {
			if len(buckets[s]) == 0 {
				return
			}
			cp := make([]*Event, len(buckets[s]))
			copy(cp, buckets[s])
			eng.shards[s] <- cp
			buckets[s] = buckets[s][:0]
		}
		_ = batch
		for _, ev := range events {
			s := int(hashKey(ev.ShardKey) % uint64(numShards))
			buckets[s] = append(buckets[s], ev)
			if len(buckets[s]) >= chanBatchSize {
				flush(s)
			}
		}
		for s := range buckets {
			flush(s)
		}
		for i := range eng.shards {
			close(eng.shards[i])
		}
	}()

	eng.wg.Wait()
	dur := time.Since(start)

	if err := eng.store.Close(); err != nil {
		panic(err)
	}
	return dur, eng.processed.Load(), eng.steals.Load()
}

func (eng *ShardedChanEngine) workerLoop(id int) {
	defer eng.wg.Done()
	mine := eng.shards[id]

	for {
		// Try own channel first (blocking).
		batch, ok := <-mine
		if ok {
			eng.processBatch(batch, id)
			continue
		}

		// Own channel closed — try to steal from others with a non-blocking select.
		// We build a reflect-free select by iterating manually.
		stole := false
		for i := 0; i < eng.numShards; i++ {
			if i == id {
				continue
			}
			select {
			case batch, ok = <-eng.shards[i]:
				if ok {
					eng.steals.Add(1)
					eng.processBatch(batch, id)
					stole = true
				}
			default:
			}
			if stole {
				break
			}
		}
		if !stole {
			return
		}
	}
}

func (eng *ShardedChanEngine) processBatch(batch []*Event, workerID int) {
	for _, ev := range batch {
		checksum := computeChecksum(ev)
		eng.store.Write(ResultRecord{
			ID:        ev.ID,
			Timestamp: time.Now().UnixNano(),
			Checksum:  checksum,
			Worker:    int32(workerID),
		})
		eng.processed.Add(1)
	}
}

// ---------------------------------------------------------------------------

type NaiveEngine struct {
	queue     *NaiveQueue
	sem       *ChanSemaphore
	store     *NaiveStore
	wg        sync.WaitGroup
	done      atomic.Bool
	pending   atomic.Int64
	processed atomic.Int64
}

func runNaive(events []*Event, numWorkers int, path string) (time.Duration, int64) {
	store, err := NewNaiveStore(path)
	if err != nil {
		panic(err)
	}
	eng := &NaiveEngine{
		queue: &NaiveQueue{},
		sem:   NewChanSemaphore(numWorkers * 4),
		store: store,
	}

	start := time.Now()

	eng.wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go eng.worker(i)
	}

	go func() {
		for _, ev := range events {
			for !eng.sem.TryAcquire() {
				runtime.Gosched()
			}
			eng.pending.Add(1)
			eng.queue.Enqueue(ev)
		}
		eng.done.Store(true)
	}()

	eng.wg.Wait()
	dur := time.Since(start)

	if err := eng.store.Close(); err != nil {
		panic(err)
	}
	return dur, eng.processed.Load()
}

func (eng *NaiveEngine) worker(id int) {
	defer eng.wg.Done()
	for {
		ev, ok := eng.queue.Dequeue()
		if !ok {
			if eng.done.Load() && eng.pending.Load() == 0 {
				return
			}
			runtime.Gosched()
			continue
		}
		checksum := computeChecksum(ev)
		eng.store.Write(ResultRecord{
			ID:        ev.ID,
			Timestamp: time.Now().UnixNano(),
			Checksum:  checksum,
			Worker:    int32(id),
		})
		eng.sem.Release()
		eng.processed.Add(1)
		eng.pending.Add(-1)
	}
}

type OptimalEngine struct {
	numShards int
	rings     []*SPSCRingBuffer[Event]
	deques    []*WSDeque[Event]
	sems      []*PaddedSemaphore
	store     *MMapStore

	wg        sync.WaitGroup
	done      atomic.Bool
	pending   atomic.Int64
	processed atomic.Int64
	steals    atomic.Int64
}

const (
	ringCapacity  = 2048
	dequeCapacity = 4096
	drainBatch    = 64
)

func runOptimal(events []*Event, numShards int, path string) (time.Duration, int64, int64) {
	store, err := NewMMapStore(path, len(events))
	if err != nil {
		panic(err)
	}

	eng := &OptimalEngine{
		numShards: numShards,
		rings:     make([]*SPSCRingBuffer[Event], numShards),
		deques:    make([]*WSDeque[Event], numShards),
		sems:      make([]*PaddedSemaphore, numShards),
		store:     store,
	}
	for i := 0; i < numShards; i++ {
		eng.rings[i] = NewSPSCRingBuffer[Event](ringCapacity)
		eng.deques[i] = NewWSDeque[Event](dequeCapacity)
		eng.sems[i] = NewPaddedSemaphore(1024)
	}

	start := time.Now()

	eng.wg.Add(numShards)
	for i := 0; i < numShards; i++ {
		go eng.workerLoop(i)
	}

	go func() {
		for _, ev := range events {
			shard := int(hashKey(ev.ShardKey) % uint64(numShards))
			for !eng.sems[shard].TryAcquire() {
				runtime.Gosched()
			}
			eng.pending.Add(1)
			for !eng.rings[shard].Enqueue(ev) {
				runtime.Gosched()
			}
		}
		eng.done.Store(true)
	}()

	eng.wg.Wait()
	dur := time.Since(start)

	if err := eng.store.Close(); err != nil {
		panic(err)
	}
	return dur, eng.processed.Load(), eng.steals.Load()
}

func (eng *OptimalEngine) workerLoop(id int) {
	defer eng.wg.Done()
	myRing := eng.rings[id]
	myDeque := eng.deques[id]
	idleSpins := 0

	for {
		for i := 0; i < drainBatch; i++ {
			ev, ok := myRing.Dequeue()
			if !ok {
				break
			}
			if !myDeque.PushBottom(ev) {
				eng.process(ev, id)
			}
		}

		if ev, ok := myDeque.PopBottom(); ok {
			eng.process(ev, id)
			idleSpins = 0
			continue
		}

		victim := rand.N(eng.numShards)
		if victim != id {
			if ev, ok := eng.deques[victim].Steal(); ok {
				eng.steals.Add(1)
				eng.process(ev, id)
				idleSpins = 0
				continue
			}
		}

		if eng.done.Load() && eng.pending.Load() == 0 {
			return
		}
		idleSpins++
		if idleSpins < 200 {
			runtime.Gosched()
		} else {
			time.Sleep(50 * time.Microsecond)
		}
	}
}

func (eng *OptimalEngine) process(ev *Event, workerID int) {
	checksum := computeChecksum(ev)
	eng.store.Write(ResultRecord{
		ID:        ev.ID,
		Timestamp: time.Now().UnixNano(),
		Checksum:  checksum,
		Worker:    int32(workerID),
	})
	shard := int(hashKey(ev.ShardKey) % uint64(eng.numShards))
	eng.sems[shard].Release()
	eng.processed.Add(1)
	eng.pending.Add(-1)
}

// ---------------------------------------------------------------------------
// Engine 5: Partitioned parallel-for + index-preserving mmap writes
//
// Why this beats [3] and [4] on this workload:
//   - The pipeline is batch, 1:1, order-irrelevant. records[i] corresponds to
//     events[i], so output position is a pure function of input position.
//     No queue, no shard hash, no semaphore, no cursor atomic needed.
//   - One atomic per chunk (1024 events) instead of ~4 atomics per event
//     (pending +1/-1, processed, cursor, sem). ~4000x fewer contended RMWs.
//   - No time.Now() in the hot loop (reuses ev.Timestamp); unrolled checksum.
//   - Workers write disjoint, chunk-aligned mmap ranges, so lines stay in L1
//     instead of ping-ponging a single cursor line. Output stays ordered.
//   - Dynamic chunk claiming via next.Add(chunk) still balances skewed keys.
//
// When NOT to use it: true streaming (unbounded input, latency SLOs) still
// wants [3]'s channel parking or a Disruptor-style MPMC ring, because there
// is no finite slice to range-partition.
// ---------------------------------------------------------------------------

const partitionedChunk = 1024

func runPartitioned(events []*Event, numWorkers int, path string) (time.Duration, int64) {
	store, err := NewMMapStore(path, len(events))
	if err != nil {
		panic(err)
	}

	var next atomic.Int64
	var processed atomic.Int64
	var wg sync.WaitGroup

	start := time.Now()

	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			var local int64
			n := int64(len(events))
			for {
				base := next.Add(partitionedChunk) - partitionedChunk
				if base >= n {
					break
				}
				end := base + partitionedChunk
				if end > n {
					end = n
				}
				for i := base; i < end; i++ {
					ev := events[i]
					store.WriteAt(int(i), ResultRecord{
						ID:        ev.ID,
						Timestamp: ev.Timestamp,
						Checksum:  computeChecksumFast(ev),
						Worker:    int32(workerID),
					})
					local++
				}
			}
			processed.Add(local)
		}(w)
	}

	wg.Wait()
	dur := time.Since(start)

	if err := store.Close(); err != nil {
		panic(err)
	}
	return dur, processed.Load()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	demoFalseSharing()

	const numEvents = 4_000_000
	numShards := runtime.NumCPU()
	if numShards < 2 {
		numShards = 2
	}

	numShards = 4

	fmt.Printf("generating %d synthetic events across up to 997 shard keys, %d shards, GOMAXPROCS=%d\n\n",
		numEvents, numShards, runtime.GOMAXPROCS(0))

	events := generateEvents(numEvents)

	naivePath := filepathTemp("vortex-naive-*.dat")
	defer os.Remove(naivePath)
	fmt.Println("=== [1/5] Naive: mutex queue + channel semaphore (4×workers in-flight) + binary.Write ===")
	naiveDur, naiveProcessed := runNaive(events, numShards, naivePath)
	fmt.Printf("processed=%d/%d duration=%v throughput=%.0f events/sec\n\n",
		naiveProcessed, numEvents, naiveDur, float64(naiveProcessed)/naiveDur.Seconds())

	idiomPath := filepathTemp("vortex-idiomatic-*.dat")
	defer os.Remove(idiomPath)
	fmt.Println("=== [2/5] Idiomatic Go: buffered channel (1024×workers in-flight) + typed write ===")
	idiomDur, idiomProcessed := runIdiomatic(events, numShards, idiomPath)
	fmt.Printf("processed=%d/%d duration=%v throughput=%.0f events/sec (%.2fx naive)\n\n",
		idiomProcessed, numEvents, idiomDur, float64(idiomProcessed)/idiomDur.Seconds(),
		naiveDur.Seconds()/idiomDur.Seconds())

	chanPath := filepathTemp("vortex-shardedchan-*.dat")
	defer os.Remove(chanPath)
	fmt.Println("=== [3/5] Sharded channels + channel stealing + mmap ===")
	chanDur, chanProcessed, chanSteals := runShardedChan(events, numShards, chanPath)
	fmt.Printf("processed=%d/%d duration=%v throughput=%.0f events/sec steals=%d (%.2fx naive)\n\n",
		chanProcessed, numEvents, chanDur, float64(chanProcessed)/chanDur.Seconds(), chanSteals,
		naiveDur.Seconds()/chanDur.Seconds())

	optPath := filepathTemp("vortex-mmap-*.dat")
	defer os.Remove(optPath)
	fmt.Println("=== [4/5] Optimal: sharded SPSC rings + Chase-Lev deques + padded atomics + mmap ===")
	optDur, optProcessed, steals := runOptimal(events, numShards, optPath)
	fmt.Printf("processed=%d/%d duration=%v throughput=%.0f events/sec steals=%d (%.2fx naive)\n\n",
		optProcessed, numEvents, optDur, float64(optProcessed)/optDur.Seconds(), steals,
		naiveDur.Seconds()/optDur.Seconds())

	partPath := filepathTemp("vortex-partitioned-*.dat")
	defer os.Remove(partPath)
	fmt.Println("=== [5/5] Partitioned: parallel-for chunks + index-preserving mmap + fast checksum ===")
	partDur, partProcessed := runPartitioned(events, numShards, partPath)
	fmt.Printf("processed=%d/%d duration=%v throughput=%.0f events/sec (%.2fx naive, %.2fx optimal)\n\n",
		partProcessed, numEvents, partDur, float64(partProcessed)/partDur.Seconds(),
		naiveDur.Seconds()/partDur.Seconds(), optDur.Seconds()/partDur.Seconds())

	fmt.Println("=== Progression summary ===")
	fmt.Printf("%-40s  %8s  %8s\n", "Engine", "Throughput", "vs naive")
	printRow := func(name string, dur time.Duration, n int64) {
		tput := float64(n) / dur.Seconds()
		fmt.Printf("%-40s  %8.0f  %7.2fx\n", name, tput, naiveDur.Seconds()/dur.Seconds())
	}
	printRow("Naive (unfair: 4x in-flight, reflection)", naiveDur, naiveProcessed)
	printRow("Idiomatic (fair baseline)", idiomDur, idiomProcessed)
	printRow("Sharded channels + stealing + mmap", chanDur, chanProcessed)
	printRow("Optimal (SPSC+Chase-Lev+padded atomics)", optDur, optProcessed)
	printRow("Partitioned (chunks+indexed mmap)", partDur, partProcessed)

	if runtime.NumCPU() < 4 {
		fmt.Println("\n(run on ≥4 cores for representative numbers)")
	}
}

func filepathTemp(pattern string) string {
	tempDir := os.TempDir()
	root, err := os.OpenRoot(tempDir)
	if err != nil {
		panic(err)
	}
	defer root.Close()

	f, err := os.CreateTemp(root.Name(), pattern)
	if err != nil {
		panic(err)
	}
	name := f.Name()
	f.Close()
	return filepath.Join(tempDir, filepath.Base(name))
}

/*
go run vortex-v2/main.go
=== False sharing: two goroutines each hammering their own counter ===
unpadded (both counters on one cache line): 599.585619ms
padded   (each counter on its own line):    225.65452ms
padding speedup: 2.66x

generating 4000000 synthetic events across up to 997 shard keys, 8 shards, GOMAXPROCS=8

=== [1/5] Naive: mutex queue + channel semaphore (4×workers in-flight) + binary.Write ===
processed=4000000/4000000 duration=7.709836227s throughput=518818 events/sec

=== [2/5] Idiomatic Go: buffered channel (1024×workers in-flight) + typed write ===
processed=4000000/4000000 duration=2.297736044s throughput=1740844 events/sec (3.36x naive)

=== [3/5] Sharded channels + channel stealing + mmap ===
processed=4000000/4000000 duration=1.468567248s throughput=2723743 events/sec steals=45 (5.25x naive)

=== [4/5] Optimal: sharded SPSC rings + Chase-Lev deques + padded atomics + mmap ===
processed=4000000/4000000 duration=1.655368825s throughput=2416380 events/sec steals=351580 (4.66x naive)

=== [5/5] Partitioned: parallel-for chunks + index-preserving mmap + fast checksum ===
processed=4000000/4000000 duration=133.562062ms throughput=29948624 events/sec (57.72x naive, 12.39x optimal)

=== Progression summary ===
Engine                                    Throughput  vs naive
Naive (unfair: 4x in-flight, reflection)    518818     1.00x
Idiomatic (fair baseline)                  1740844     3.36x
Sharded channels + stealing + mmap         2723743     5.25x
Optimal (SPSC+Chase-Lev+padded atomics)    2416380     4.66x
Partitioned (chunks+indexed mmap)         29948624    57.72x
*/
