package sketches

import (
	"bytes"
	"fmt"
	"math"
)

// cmMaxCells caps w*d so a constructor or a decoder cannot be asked to allocate an
// unbounded number of counters. 1<<28 counters is 2 GiB of uint64, far above any
// real configuration and well below an overflow of int.
const cmMaxCells = 1 << 28

// CountMin is a count-min sketch: it estimates the frequency of a key in a stream
// and never underestimates. With width w and depth d it guarantees, for every key
// x, f(x) <= Estimate(x) <= f(x) + epsilon*N with probability at least 1-delta,
// where N is the total weight added and epsilon, delta are the parameters given to
// NewCountMin. The d rows use one base hash and a deterministic double-hashing
// derivation (row r uses h1 + r*h2), so no per-row seed is stored and a decoded
// sketch uses the same hash family. Not safe for concurrent use.
type CountMin struct {
	w, d    uint32
	epsilon float64
	delta   float64
	counts  []uint64
	total   uint64
}

// NewCountMin returns a count-min sketch sized for additive error epsilon (as a
// fraction of the total weight) with failure probability at most delta. It sets
// w = ceil(e/epsilon) and d = ceil(ln(1/delta)). Both epsilon and delta must be in
// the open interval (0,1), and the resulting table must not exceed an internal cap
// on the number of counters; otherwise it returns an error wrapping ErrInvalidParam.
// cmDims validates the parameters and derives the table shape without allocating,
// so a decoder can bound the implied allocation before it commits any memory.
func cmDims(epsilon, delta float64) (w, d uint32, err error) {
	if !(epsilon > 0 && epsilon < 1) || math.IsNaN(epsilon) {
		return 0, 0, fmt.Errorf("%w: epsilon must be in (0,1), got %g", ErrInvalidParam, epsilon)
	}
	if !(delta > 0 && delta < 1) || math.IsNaN(delta) {
		return 0, 0, fmt.Errorf("%w: delta must be in (0,1), got %g", ErrInvalidParam, delta)
	}
	w = uint32(math.Ceil(math.E / epsilon))
	d = uint32(math.Ceil(math.Log(1 / delta)))
	if w < 1 {
		w = 1
	}
	if d < 1 {
		d = 1
	}
	if uint64(w)*uint64(d) > cmMaxCells {
		return 0, 0, fmt.Errorf("%w: epsilon=%g delta=%g would need %d counters, over the %d cap", ErrInvalidParam, epsilon, delta, uint64(w)*uint64(d), cmMaxCells)
	}
	return w, d, nil
}

func NewCountMin(epsilon, delta float64) (*CountMin, error) {
	w, d, err := cmDims(epsilon, delta)
	if err != nil {
		return nil, err
	}
	return &CountMin{w: w, d: d, epsilon: epsilon, delta: delta, counts: make([]uint64, uint64(w)*uint64(d))}, nil
}

// Width and Depth report the table shape.
func (c *CountMin) Width() int { return int(c.w) }
func (c *CountMin) Depth() int { return int(c.d) }

// Total is the sum of all weights added.
func (c *CountMin) Total() uint64 { return c.total }

func (c *CountMin) cols(base uint64) func(r uint32) uint32 {
	h1 := base
	h2 := mix64(base) | 1
	return func(r uint32) uint32 {
		return uint32((h1 + uint64(r)*h2) % uint64(c.w))
	}
}

// AddHash adds weight count to the key identified by the precomputed hash base. It
// is the hash door for callers who bring their own 64-bit hash; Add hashes the key
// with the package's fixed mixer. A count of zero is a no-op.
func (c *CountMin) AddHash(base uint64, count uint64) {
	if count == 0 {
		return
	}
	col := c.cols(base)
	for r := uint32(0); r < c.d; r++ {
		c.counts[r*c.w+col(r)] += count
	}
	c.total += count
}

// Add adds weight count to key. A count of zero is a no-op.
func (c *CountMin) Add(key []byte, count uint64) { c.AddHash(hashBytes(key), count) }

// EstimateHash returns the estimated frequency of the key identified by base.
func (c *CountMin) EstimateHash(base uint64) uint64 {
	col := c.cols(base)
	var est uint64
	for r := uint32(0); r < c.d; r++ {
		v := c.counts[r*c.w+col(r)]
		if r == 0 || v < est {
			est = v
		}
	}
	return est
}

// Estimate returns the estimated frequency of key. It never underestimates the
// true count.
func (c *CountMin) Estimate(key []byte) uint64 { return c.EstimateHash(hashBytes(key)) }

// Merge adds other into c. The two sketches must have the same parameters; a
// mismatch returns an error wrapping ErrIncompatibleMerge.
func (c *CountMin) Merge(other *CountMin) error {
	if other.w != c.w || other.d != c.d {
		return fmt.Errorf("%w: count-min shapes %dx%d and %dx%d differ", ErrIncompatibleMerge, c.d, c.w, other.d, other.w)
	}
	for i := range c.counts {
		c.counts[i] += other.counts[i]
	}
	c.total += other.total
	return nil
}

const cmMagic0, cmMagic1, cmVersion = 'C', 'M', 1

// Encode returns the binary form of the sketch. Decode reverses it.
func (c *CountMin) Encode() []byte {
	var buf bytes.Buffer
	buf.WriteByte(cmMagic0)
	buf.WriteByte(cmMagic1)
	buf.WriteByte(cmVersion)
	putFloat(&buf, c.epsilon)
	putFloat(&buf, c.delta)
	putUvarint(&buf, c.total)
	for _, v := range c.counts {
		putUvarint(&buf, v)
	}
	return buf.Bytes()
}

// DecodeCountMin parses the output of Encode. It derives w and d from the stored
// epsilon and delta rather than trusting a stored shape, and rejects a buffer whose
// remaining bytes cannot hold the w*d counters, so a malformed length cannot force
// a large allocation. It returns an error wrapping ErrBadData or
// ErrUnsupportedVersion on any problem.
func DecodeCountMin(data []byte) (*CountMin, error) {
	r := &reader{b: data}
	m0, err := r.readByte()
	if err != nil {
		return nil, err
	}
	m1, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if m0 != cmMagic0 || m1 != cmMagic1 {
		return nil, fmt.Errorf("%w: not a count-min sketch", ErrBadData)
	}
	ver, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if ver != cmVersion {
		return nil, fmt.Errorf("%w: count-min version %d", ErrUnsupportedVersion, ver)
	}
	epsilon, err := r.readFloat()
	if err != nil {
		return nil, err
	}
	delta, err := r.readFloat()
	if err != nil {
		return nil, err
	}
	w, d, err := cmDims(epsilon, delta)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadData, err)
	}
	// Bound the implied allocation by the remaining bytes before committing any
	// memory: each counter needs at least one encoded byte, so a table larger than
	// what is left cannot be genuine.
	if uint64(w)*uint64(d) > uint64(r.remaining()) {
		return nil, fmt.Errorf("%w: %dx%d table needs more bytes than the %d remaining", ErrBadData, d, w, r.remaining())
	}
	total, err := r.readUvarint()
	if err != nil {
		return nil, err
	}
	c := &CountMin{w: w, d: d, epsilon: epsilon, delta: delta, counts: make([]uint64, uint64(w)*uint64(d))}
	cells := len(c.counts)
	for i := 0; i < cells; i++ {
		v, err := r.readUvarint()
		if err != nil {
			return nil, err
		}
		c.counts[i] = v
	}
	c.total = total
	if !r.atEnd() {
		return nil, fmt.Errorf("%w: %d trailing bytes", ErrBadData, r.remaining())
	}
	return c, nil
}
