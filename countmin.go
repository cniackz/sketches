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
// NewCountMin. Each of the d rows derives an independent column by running the fixed
// mixer over the key's base hash combined with a distinct per-row constant, so no
// per-row seed is stored and a decoded sketch uses the same hash family. Not safe for
// concurrent use.
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

// cols returns, for a key's base hash, the column it falls in on each of the d rows.
// Each row runs the fixed mixer over the base XOR a distinct per-row constant, so the
// rows are independent. An earlier double-hashing scheme (h1 + r*h2) shared h1 and h2
// across every row, which made two keys collide in ALL d rows with probability about
// 1/w^2 instead of 1/w^d, and broke the 1-delta bound when w was small and d large.
// The per-row constants are derived from r, not stored, so a decoded sketch uses the
// same hash family.
func (c *CountMin) cols(base uint64) func(r uint32) uint32 {
	return func(r uint32) uint32 {
		h := mix64(base ^ ((uint64(r) + 1) * 0x9e3779b97f4a7c15))
		return uint32(h % uint64(c.w))
	}
}

// satAdd64 adds two counts, saturating at the maximum rather than wrapping, so a count
// large enough to overflow uint64 cannot make an estimate fall below the true count.
func satAdd64(a, b uint64) uint64 {
	if a > math.MaxUint64-b {
		return math.MaxUint64
	}
	return a + b
}

// AddHash adds weight count to the key identified by the precomputed hash base. It
// is the hash door for callers who bring their own 64-bit hash, which should be
// well distributed (the mixer is applied per row, but a base with little entropy
// gives correlated rows); Add hashes the key with the package's fixed mixer. A count
// of zero, or a call on a zero-value sketch, is a no-op.
func (c *CountMin) AddHash(base uint64, count uint64) {
	if count == 0 || c.w == 0 {
		return
	}
	col := c.cols(base)
	for r := uint32(0); r < c.d; r++ {
		i := r*c.w + col(r)
		c.counts[i] = satAdd64(c.counts[i], count)
	}
	c.total = satAdd64(c.total, count)
}

// Add adds weight count to key. A count of zero is a no-op.
func (c *CountMin) Add(key []byte, count uint64) { c.AddHash(hashBytes(key), count) }

// EstimateHash returns the estimated frequency of the key identified by base.
func (c *CountMin) EstimateHash(base uint64) uint64 {
	if c.w == 0 {
		return 0
	}
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
	if other == nil {
		return fmt.Errorf("%w: cannot merge a nil count-min sketch", ErrInvalidInput)
	}
	if other.w != c.w || other.d != c.d {
		return fmt.Errorf("%w: count-min shapes %dx%d and %dx%d differ", ErrIncompatibleMerge, c.d, c.w, other.d, other.w)
	}
	for i := range c.counts {
		c.counts[i] = satAdd64(c.counts[i], other.counts[i])
	}
	c.total = satAdd64(c.total, other.total)
	return nil
}

// cmVersion is 2: version 1 used a correlated double-hashing scheme (h1 + r*h2), so a
// v1 blob decoded under the current independent per-row hash would read its counters
// from the wrong columns and underestimate. Decode rejects version 1 rather than
// reading it wrong.
const cmMagic0, cmMagic1, cmVersion = 'C', 'M', 2

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
	// Every add touches exactly one cell per row with the same count, so each row
	// receives the same multiset of counts as the total, and its saturating sum must
	// equal the stored total. The sum uses satAdd64, the same saturation Add uses, so a
	// sketch that saturated a cell still round trips; a row whose saturating sum does
	// not match the total is a forged or corrupt sketch and is rejected.
	for rw := uint32(0); rw < d; rw++ {
		var rowSum uint64
		for j := uint32(0); j < w; j++ {
			rowSum = satAdd64(rowSum, c.counts[rw*w+j])
		}
		if rowSum != total {
			return nil, fmt.Errorf("%w: count-min row %d sums to %d, stored total %d", ErrBadData, rw, rowSum, total)
		}
	}
	return c, nil
}
