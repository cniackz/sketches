package sketches

import (
	"bytes"
	"fmt"
	"math"
	"sort"
)

// TDigest estimates quantiles of a stream of real values with good accuracy,
// especially at the extreme quantiles, and is fully mergeable. Unlike DDSketch it
// carries NO worst-case error guarantee: its accuracy is empirical, controlled by
// the compression parameter, and best near q=0 and q=1 because its scale function
// makes the clusters smaller in the tails. Use DDSketch when a provable bound is
// required, and TDigest when tail accuracy at a small size matters more. Not safe
// for concurrent use. Clean-room from Dunning and Ertl, the merging variant.
type TDigest struct {
	compression float64
	merged      []centroid
	buffer      []centroid
	bufferLimit int
	totalWeight float64
	bufWeight   float64 // weight held in buffer, kept current so Count and the overflow guard are O(1)
	min, max    float64
}

type centroid struct {
	mean, weight float64
}

// NewTDigest returns an empty digest. compression must be greater than 1; larger
// values give more accuracy and more centroids. 100 is a common choice.
func NewTDigest(compression float64) (*TDigest, error) {
	if !(compression > 1) || math.IsInf(compression, 0) {
		return nil, fmt.Errorf("%w: compression must be > 1, got %g", ErrInvalidParam, compression)
	}
	limit := int(10 * compression)
	if limit < 50 {
		limit = 50
	}
	return &TDigest{compression: compression, bufferLimit: limit, min: math.Inf(1), max: math.Inf(-1)}, nil
}

// Add records value v. A NaN or infinite value is rejected with an error wrapping
// ErrInvalidInput and leaves the digest unchanged.
func (t *TDigest) Add(v float64) error { return t.AddWeighted(v, 1) }

// AddWeighted records value v with weight w (w > 0). A NaN or infinite v, a
// non-positive or non-finite w, or a w that would make the running total weight
// overflow to infinity is rejected with an error wrapping ErrInvalidInput and leaves
// the digest unchanged.
func (t *TDigest) AddWeighted(v, w float64) error {
	if t.compression == 0 {
		return fmt.Errorf("%w: use NewTDigest to create a TDigest", ErrInvalidInput)
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%w: cannot add %g to a TDigest", ErrInvalidInput, v)
	}
	if !(w > 0) || math.IsInf(w, 0) || math.IsNaN(w) {
		return fmt.Errorf("%w: weight must be a positive finite number, got %g", ErrInvalidInput, w)
	}
	if math.IsInf(t.totalWeight+t.bufWeight+w, 0) {
		return fmt.Errorf("%w: adding weight %g would overflow the total weight", ErrInvalidInput, w)
	}
	t.buffer = append(t.buffer, centroid{v, w})
	t.bufWeight += w
	if v < t.min {
		t.min = v
	}
	if v > t.max {
		t.max = v
	}
	if len(t.buffer) >= t.bufferLimit {
		t.compress()
	}
	return nil
}

func clamp01(q float64) float64 {
	if q < 0 {
		return 0
	}
	if q > 1 {
		return 1
	}
	return q
}

func kScale(q, comp float64) float64 { return comp / (2 * math.Pi) * math.Asin(2*clamp01(q)-1) }
func qScale(k, comp float64) float64 { return clamp01((math.Sin(k*2*math.Pi/comp) + 1) / 2) }

// compress merges the buffered points into the sorted centroid list using the
// scale function, bounding each centroid to about one k-unit of quantile width.
func (t *TDigest) compress() {
	if len(t.buffer) == 0 {
		return
	}
	all := append(t.merged, t.buffer...)
	t.buffer = t.buffer[:0]
	t.bufWeight = 0
	sort.Slice(all, func(i, j int) bool { return all[i].mean < all[j].mean })
	var total float64
	for _, c := range all {
		total += c.weight
	}
	out := make([]centroid, 1, len(all))
	out[0] = all[0]
	cumulative := all[0].weight
	qLimitWT := qScale(kScale(0, t.compression)+1, t.compression) * total
	for i := 1; i < len(all); i++ {
		c := all[i]
		if cumulative+c.weight <= qLimitWT {
			last := &out[len(out)-1]
			last.weight += c.weight
			last.mean += (c.mean - last.mean) * c.weight / last.weight
			cumulative += c.weight
		} else {
			qLeft := cumulative / total
			qLimitWT = qScale(kScale(qLeft, t.compression)+1, t.compression) * total
			out = append(out, c)
			cumulative += c.weight
		}
	}
	t.merged = out
	t.totalWeight = total
}

// Compression, Count and NumCentroids report the configured compression, the total
// weight, and the number of stored centroids.
func (t *TDigest) Compression() float64 { return t.compression }
func (t *TDigest) Count() float64       { return t.totalWeight + t.bufWeight }
func (t *TDigest) NumCentroids() int {
	t.compress()
	return len(t.merged)
}

// lerp interpolates linearly as a convex combination, y0*(1-u) + y1*u with u in
// [0,1], rather than y0 + (y1-y0)*(x-x0)/(x1-x0). The convex form keeps each term
// bounded by |y0| or |y1|, so with finite endpoints the result is always finite and
// within [min(y0,y1), max(y0,y1)]; the subtractive form could overflow to Inf for a
// decoded digest with an extreme min or max and a huge centroid weight.
func lerp(x0, y0, x1, y1, x float64) float64 {
	if x1 == x0 {
		return y0/2 + y1/2
	}
	u := (x - x0) / (x1 - x0)
	return y0*(1-u) + y1*u
}

// Quantile returns the estimated value at quantile q in [0,1]. It returns an error
// wrapping ErrInvalidInput for a q outside [0,1] or NaN, and ErrEmpty for a digest
// with no values. The estimate is monotone non-decreasing in q.
func (t *TDigest) Quantile(q float64) (float64, error) {
	if math.IsNaN(q) || q < 0 || q > 1 {
		return 0, fmt.Errorf("%w: quantile q=%g outside [0,1]", ErrInvalidInput, q)
	}
	t.compress()
	if t.totalWeight == 0 {
		return 0, ErrEmpty
	}
	if q <= 0 {
		return t.min, nil
	}
	if q >= 1 {
		return t.max, nil
	}
	target := q * t.totalWeight
	var cum float64
	for i, c := range t.merged {
		center := cum + c.weight/2
		if target <= center {
			if i == 0 {
				return lerp(0, t.min, center, c.mean, target), nil
			}
			prev := t.merged[i-1]
			prevCenter := cum - prev.weight/2
			return lerp(prevCenter, prev.mean, center, c.mean, target), nil
		}
		cum += c.weight
	}
	last := t.merged[len(t.merged)-1]
	lastCenter := t.totalWeight - last.weight/2
	return lerp(lastCenter, last.mean, t.totalWeight, t.max, target), nil
}

// Merge folds other into t. Both must have the same compression; a mismatch
// returns an error wrapping ErrIncompatibleMerge.
func (t *TDigest) Merge(other *TDigest) error {
	if other == nil {
		return fmt.Errorf("%w: cannot merge a nil TDigest", ErrInvalidInput)
	}
	if other.compression != t.compression {
		return fmt.Errorf("%w: TDigest compressions %g and %g differ", ErrIncompatibleMerge, t.compression, other.compression)
	}
	if math.IsInf(t.Count()+other.Count(), 0) {
		return fmt.Errorf("%w: merging would overflow the total weight", ErrInvalidInput)
	}
	t.buffer = append(t.buffer, other.merged...)
	t.buffer = append(t.buffer, other.buffer...)
	t.bufWeight += other.Count()
	if other.min < t.min {
		t.min = other.min
	}
	if other.max > t.max {
		t.max = other.max
	}
	t.compress()
	return nil
}

const (
	tdMagic0, tdMagic1, tdVersion = 'T', 'D', 1
)

// Encode returns the binary form. It compresses first, so the centroid count is
// bounded by the compression. Decode reverses it.
func (t *TDigest) Encode() []byte {
	t.compress()
	var buf bytes.Buffer
	buf.WriteByte(tdMagic0)
	buf.WriteByte(tdMagic1)
	buf.WriteByte(tdVersion)
	putFloat(&buf, t.compression)
	putFloat(&buf, t.min)
	putFloat(&buf, t.max)
	putUvarint(&buf, uint64(len(t.merged)))
	for _, c := range t.merged {
		putFloat(&buf, c.mean)
		putFloat(&buf, c.weight)
	}
	return buf.Bytes()
}

// DecodeTDigest parses the output of Encode. It bounds the centroid count by the
// compression and the remaining bytes, and requires the means to be sorted and the
// weights positive and finite, so a malformed length or a bad centroid cannot force
// a large allocation or a non-monotone digest. It returns an error wrapping
// ErrBadData or ErrUnsupportedVersion.
func DecodeTDigest(data []byte) (*TDigest, error) {
	r := &reader{b: data}
	m0, err := r.readByte()
	if err != nil {
		return nil, err
	}
	m1, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if m0 != tdMagic0 || m1 != tdMagic1 {
		return nil, fmt.Errorf("%w: not a TDigest", ErrBadData)
	}
	ver, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if ver != tdVersion {
		return nil, fmt.Errorf("%w: TDigest version %d", ErrUnsupportedVersion, ver)
	}
	compression, err := r.readFloat()
	if err != nil {
		return nil, err
	}
	t, err := NewTDigest(compression)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadData, err)
	}
	mn, err := r.readFloat()
	if err != nil {
		return nil, err
	}
	mx, err := r.readFloat()
	if err != nil {
		return nil, err
	}
	n, err := r.readUvarint()
	if err != nil {
		return nil, err
	}
	// Each centroid needs 16 bytes, so the count cannot exceed remaining/16; the
	// division avoids an overflow of n*16, and this bounds the allocation before any
	// slice is made.
	if n > uint64(r.remaining())/16 {
		return nil, fmt.Errorf("%w: centroid count %d needs more bytes than the %d remaining", ErrBadData, n, r.remaining())
	}
	cs := make([]centroid, 0, n)
	var total, prevMean float64
	for i := uint64(0); i < n; i++ {
		mean, err := r.readFloat()
		if err != nil {
			return nil, err
		}
		w, err := r.readFloat()
		if err != nil {
			return nil, err
		}
		if math.IsNaN(mean) || math.IsInf(mean, 0) {
			return nil, fmt.Errorf("%w: non-finite centroid mean", ErrBadData)
		}
		if !(w > 0) || math.IsInf(w, 0) || math.IsNaN(w) {
			return nil, fmt.Errorf("%w: non-positive centroid weight", ErrBadData)
		}
		if i > 0 && mean < prevMean {
			return nil, fmt.Errorf("%w: centroid means not sorted", ErrBadData)
		}
		prevMean = mean
		cs = append(cs, centroid{mean, w})
		total += w
	}
	if !r.atEnd() {
		return nil, fmt.Errorf("%w: %d trailing bytes", ErrBadData, r.remaining())
	}
	t.merged = cs
	t.totalWeight = total
	if n > 0 {
		if math.IsNaN(mn) || math.IsInf(mn, 0) || math.IsNaN(mx) || math.IsInf(mx, 0) || mn > mx {
			return nil, fmt.Errorf("%w: min/max must be finite with min <= max", ErrBadData)
		}
		if cs[0].mean < mn || cs[len(cs)-1].mean > mx {
			return nil, fmt.Errorf("%w: centroid means outside [min,max]", ErrBadData)
		}
		t.min = mn
		t.max = mx
	} else {
		// An empty digest carries no real bounds; normalize to the empty sentinels so a
		// forged min/max (NaN, Inf) cannot poison a later Add or Quantile.
		t.min = math.Inf(1)
		t.max = math.Inf(-1)
	}
	return t, nil
}
