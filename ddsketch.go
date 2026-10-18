package sketches

import (
	"bytes"
	"fmt"
	"math"
	"slices"
)

// DDSketch is a quantile sketch with a relative-error guarantee. For a relative
// accuracy alpha in (0,1), every value it returns for a quantile q is within alpha
// relative error of the true q-quantile of the data: |Quantile(q) - x_q| <=
// alpha*|x_q|. It handles all finite reals through a logarithmic bucketing of the
// magnitude, with a mirror store for negatives and a separate counter for zero.
// Memory grows with the logarithm of the data range, not with the count. An
// optional bucket cap (off by default) bounds memory by collapsing the
// lowest-magnitude buckets, which relaxes the guarantee for those values only.
// Not safe for concurrent use. Clean-room from Masson, Rong and Nguyen 2019.
type DDSketch struct {
	alpha      float64
	gamma      float64
	logGamma   float64
	vmul       float64 // 2/(gamma+1), the bucket value multiplier
	pos        map[int]uint64
	neg        map[int]uint64
	zeroCount  uint64
	count      uint64
	maxBuckets int // 0 means unbounded
}

func newDD(alpha float64, maxBuckets int) (*DDSketch, error) {
	if !(alpha > 0 && alpha < 1) || math.IsNaN(alpha) {
		return nil, fmt.Errorf("%w: alpha must be in (0,1), got %g", ErrInvalidParam, alpha)
	}
	if maxBuckets < 0 {
		return nil, fmt.Errorf("%w: maxBuckets must be >= 0, got %d", ErrInvalidParam, maxBuckets)
	}
	gamma := (1 + alpha) / (1 - alpha)
	return &DDSketch{
		alpha:      alpha,
		gamma:      gamma,
		logGamma:   math.Log(gamma),
		vmul:       2 / (gamma + 1),
		pos:        map[int]uint64{},
		neg:        map[int]uint64{},
		maxBuckets: maxBuckets,
	}, nil
}

// NewDDSketch returns an unbounded sketch with relative accuracy alpha in (0,1).
func NewDDSketch(alpha float64) (*DDSketch, error) { return newDD(alpha, 0) }

// NewDDSketchBounded returns a sketch capped at maxBuckets positive and maxBuckets
// negative buckets. When the cap is reached the lowest-magnitude buckets collapse
// into the boundary, so the relative-error guarantee holds for every value except
// those below the collapsed boundary. maxBuckets must be at least 1.
func NewDDSketchBounded(alpha float64, maxBuckets int) (*DDSketch, error) {
	if maxBuckets < 1 {
		return nil, fmt.Errorf("%w: bounded sketch needs maxBuckets >= 1, got %d", ErrInvalidParam, maxBuckets)
	}
	return newDD(alpha, maxBuckets)
}

// Alpha is the relative accuracy. Count is the number of values added. NumBuckets
// is the number of non-empty magnitude buckets, a proxy for memory.
func (s *DDSketch) Alpha() float64 { return s.alpha }
func (s *DDSketch) Count() uint64  { return s.count }
func (s *DDSketch) NumBuckets() int {
	n := len(s.pos) + len(s.neg)
	if s.zeroCount > 0 {
		n++
	}
	return n
}

func (s *DDSketch) index(magnitude float64) int {
	return int(math.Ceil(math.Log(magnitude) / s.logGamma))
}

func (s *DDSketch) value(i int) float64 { return math.Exp(float64(i)*s.logGamma) * s.vmul }

// Add records a value. A NaN or infinite value is rejected with an error wrapping
// ErrInvalidInput and leaves the sketch unchanged.
func (s *DDSketch) Add(v float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%w: cannot add %g to a DDSketch", ErrInvalidInput, v)
	}
	switch {
	case v == 0:
		s.zeroCount++
	case v > 0:
		s.pos[s.index(v)]++
		s.collapse(s.pos)
	default:
		s.neg[s.index(-v)]++
		s.collapse(s.neg)
	}
	s.count++
	return nil
}

// collapse merges the lowest-index buckets into the boundary when a bounded sketch
// is over its cap, so memory stays bounded at the cost of the guarantee for the
// smallest magnitudes.
func (s *DDSketch) collapse(buckets map[int]uint64) {
	if s.maxBuckets <= 0 || len(buckets) <= s.maxBuckets {
		return
	}
	idxs := make([]int, 0, len(buckets))
	for i := range buckets {
		idxs = append(idxs, i)
	}
	slices.Sort(idxs)
	nCollapse := len(idxs) - s.maxBuckets
	target := idxs[nCollapse]
	var moved uint64
	for i := 0; i < nCollapse; i++ {
		moved += buckets[idxs[i]]
		delete(buckets, idxs[i])
	}
	buckets[target] += moved
}

// Quantile returns the value at quantile q in [0,1]. It returns an error wrapping
// ErrInvalidInput for a q outside [0,1] or NaN, and ErrEmpty for a sketch with no
// values.
func (s *DDSketch) Quantile(q float64) (float64, error) {
	if math.IsNaN(q) || q < 0 || q > 1 {
		return 0, fmt.Errorf("%w: quantile q=%g outside [0,1]", ErrInvalidInput, q)
	}
	if s.count == 0 {
		return 0, ErrEmpty
	}
	rank := q * float64(s.count-1)
	var cum float64
	// Negatives in ascending value order: largest magnitude (largest index) first.
	negIdx := sortedKeys(s.neg)
	for i := len(negIdx) - 1; i >= 0; i-- {
		cum += float64(s.neg[negIdx[i]])
		if cum > rank {
			return -s.value(negIdx[i]), nil
		}
	}
	cum += float64(s.zeroCount)
	if cum > rank {
		return 0, nil
	}
	posIdx := sortedKeys(s.pos)
	for _, i := range posIdx {
		cum += float64(s.pos[i])
		if cum > rank {
			return s.value(i), nil
		}
	}
	// Unreachable for count >= 1, since cum ends at count > rank; return the max.
	return s.value(posIdx[len(posIdx)-1]), nil
}

// Merge folds other into s. Both must have the same alpha; a mismatch returns an
// error wrapping ErrIncompatibleMerge. The receiver's bucket cap, if any, is
// applied after merging.
func (s *DDSketch) Merge(other *DDSketch) error {
	if other.alpha != s.alpha {
		return fmt.Errorf("%w: DDSketch alphas %g and %g differ", ErrIncompatibleMerge, s.alpha, other.alpha)
	}
	for i, c := range other.pos {
		s.pos[i] += c
	}
	for i, c := range other.neg {
		s.neg[i] += c
	}
	s.zeroCount += other.zeroCount
	s.count += other.count
	s.collapse(s.pos)
	s.collapse(s.neg)
	return nil
}

func sortedKeys(m map[int]uint64) []int {
	ks := make([]int, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks
}

const (
	ddMagic0, ddMagic1, ddVersion = 'D', 'D', 1
)

// Encode returns the binary form. Decode reverses it.
func (s *DDSketch) Encode() []byte {
	var buf bytes.Buffer
	buf.WriteByte(ddMagic0)
	buf.WriteByte(ddMagic1)
	buf.WriteByte(ddVersion)
	putFloat(&buf, s.alpha)
	putUvarint(&buf, uint64(s.maxBuckets))
	putUvarint(&buf, s.zeroCount)
	putUvarint(&buf, s.count)
	writeBuckets(&buf, s.pos)
	writeBuckets(&buf, s.neg)
	return buf.Bytes()
}

func writeBuckets(buf *bytes.Buffer, m map[int]uint64) {
	idxs := sortedKeys(m)
	putUvarint(buf, uint64(len(idxs)))
	for _, i := range idxs {
		putZigzag(buf, int64(i))
		putUvarint(buf, m[i])
	}
}

// maxPossibleBuckets is the most distinct magnitude buckets the float64 range can
// produce at this gamma, used to cap a decoder's allocation by the parameters
// rather than only by the byte length.
func maxPossibleBuckets(logGamma float64) int {
	hi := math.Ceil(math.Log(math.MaxFloat64) / logGamma)
	lo := math.Floor(math.Log(math.SmallestNonzeroFloat64) / logGamma)
	return int(hi-lo) + 2
}

// DecodeDDSketch parses the output of Encode. It bounds each bucket count by the
// most buckets the alpha allows over the float64 range and by the remaining bytes,
// rejects a duplicate index, and checks that the bucket counts and the zero count
// sum to the stored total, so a malformed length or count cannot force a large
// allocation or a wrong sketch. It returns an error wrapping ErrBadData or
// ErrUnsupportedVersion.
func DecodeDDSketch(data []byte) (*DDSketch, error) {
	r := &reader{b: data}
	m0, err := r.readByte()
	if err != nil {
		return nil, err
	}
	m1, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if m0 != ddMagic0 || m1 != ddMagic1 {
		return nil, fmt.Errorf("%w: not a DDSketch", ErrBadData)
	}
	ver, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if ver != ddVersion {
		return nil, fmt.Errorf("%w: DDSketch version %d", ErrUnsupportedVersion, ver)
	}
	alpha, err := r.readFloat()
	if err != nil {
		return nil, err
	}
	maxBucketsU, err := r.readUvarint()
	if err != nil {
		return nil, err
	}
	if maxBucketsU > uint64(maxUint31) {
		return nil, fmt.Errorf("%w: maxBuckets %d too large", ErrBadData, maxBucketsU)
	}
	s, err := newDD(alpha, int(maxBucketsU))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadData, err)
	}
	zeroCount, err := r.readUvarint()
	if err != nil {
		return nil, err
	}
	total, err := r.readUvarint()
	if err != nil {
		return nil, err
	}
	bucketCap := maxPossibleBuckets(s.logGamma)
	sum := zeroCount
	posSum, err := readBuckets(r, s.pos, bucketCap)
	if err != nil {
		return nil, err
	}
	negSum, err := readBuckets(r, s.neg, bucketCap)
	if err != nil {
		return nil, err
	}
	sum += posSum + negSum
	if sum != total {
		return nil, fmt.Errorf("%w: bucket counts sum to %d, stored total %d", ErrBadData, sum, total)
	}
	s.zeroCount = zeroCount
	s.count = total
	if !r.atEnd() {
		return nil, fmt.Errorf("%w: %d trailing bytes", ErrBadData, r.remaining())
	}
	return s, nil
}

const maxUint31 = 1<<31 - 1

func readBuckets(r *reader, m map[int]uint64, maxCap int) (uint64, error) {
	n, err := r.readUvarint()
	if err != nil {
		return 0, err
	}
	if n > uint64(maxCap) || n > uint64(r.remaining()) {
		return 0, fmt.Errorf("%w: bucket count %d over bound", ErrBadData, n)
	}
	var sum uint64
	for i := uint64(0); i < n; i++ {
		idx, err := r.readZigzag()
		if err != nil {
			return 0, err
		}
		c, err := r.readUvarint()
		if err != nil {
			return 0, err
		}
		if c == 0 {
			return 0, fmt.Errorf("%w: zero-count bucket", ErrBadData)
		}
		if _, dup := m[int(idx)]; dup {
			return 0, fmt.Errorf("%w: duplicate bucket index %d", ErrBadData, idx)
		}
		m[int(idx)] = c
		sum += c
	}
	return sum, nil
}
