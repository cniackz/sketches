package sketches

import (
	"bytes"
	"fmt"
	"math"
	"math/bits"
	"slices"
)

const (
	hllMinPrecision = 4
	hllMaxPrecision = 18
)

// HLL is a HyperLogLog cardinality estimator with 64-bit hashing. With precision p
// it uses m = 2^p registers and estimates the number of distinct items with a
// relative standard error of about 1.04/sqrt(m); the error is statistical, not a
// per-estimate bound. At low cardinality it keeps a sparse representation and
// switches to linear counting, which removes the bias HyperLogLog has when most
// registers are still empty. A sketch is stored sparse or dense; the format byte
// records which. Not safe for concurrent use. Clean-room from Flajolet, Fusy,
// Gandon and Meunier 2007 and Heule, Nunkesser and Hall 2013; the empirical
// bias-correction tables of HLL++ are not reproduced, linear counting covers the
// small-cardinality regime instead.
//
// Because those bias-correction tables are omitted, there is one place where the
// mean error rises above 1.04/sqrt(m): the transition band just above where linear
// counting hands off to the raw estimate, roughly 2.5*m to 3*m distinct items, where
// the raw HyperLogLog estimate is positively biased by up to a few percent (measured
// about +2.4% at p=14, n=2.5*m). This is a known limit of the raw estimator, not a
// bug; outside that band the error returns to about 1.04/sqrt(m). Reproducing the
// HLL++ bias tables, or switching to Ertl's maximum-likelihood estimator, would close
// the gap; both are out of scope for this version.
type HLL struct {
	p         uint8
	m         uint32
	dense     []uint8
	sparse    map[uint32]uint8
	sparseMax int
}

// NewHLL returns an empty estimator with the given precision, which must be in
// [4,18]; otherwise it returns an error wrapping ErrInvalidParam. Precision 14
// (m=16384, standard error about 0.81%) is a good default.
func NewHLL(precision uint8) (*HLL, error) {
	if precision < hllMinPrecision || precision > hllMaxPrecision {
		return nil, fmt.Errorf("%w: HLL precision must be in [%d,%d], got %d", ErrInvalidParam, hllMinPrecision, hllMaxPrecision, precision)
	}
	m := uint32(1) << precision
	return &HLL{p: precision, m: m, sparse: map[uint32]uint8{}, sparseMax: int(m) / 4}, nil
}

// Precision reports p.
func (h *HLL) Precision() int { return int(h.p) }

func (h *HLL) indexRank(x uint64) (uint32, uint8) {
	idx := uint32(x >> (64 - h.p))
	rest := (x << h.p) | (1 << (h.p - 1))
	rank := uint8(bits.LeadingZeros64(rest) + 1)
	return idx, rank
}

// AddHash records the item identified by the precomputed 64-bit hash x. It is the
// hash door for callers who bring their own hash; Add hashes the bytes with the
// package's fixed mixer.
func (h *HLL) AddHash(x uint64) {
	idx, rank := h.indexRank(x)
	if h.dense != nil {
		if rank > h.dense[idx] {
			h.dense[idx] = rank
		}
		return
	}
	if cur, ok := h.sparse[idx]; ok {
		if rank > cur {
			h.sparse[idx] = rank
		}
		return
	}
	if len(h.sparse) >= h.sparseMax {
		h.densify()
		if rank > h.dense[idx] {
			h.dense[idx] = rank
		}
		return
	}
	h.sparse[idx] = rank
}

// Add records an item.
func (h *HLL) Add(item []byte) { h.AddHash(hashBytes(item)) }

func (h *HLL) densify() {
	h.dense = make([]uint8, h.m)
	for idx, rank := range h.sparse {
		h.dense[idx] = rank
	}
	h.sparse = nil
}

func alphaM(m uint32) float64 {
	switch m {
	case 16:
		return 0.673
	case 32:
		return 0.697
	case 64:
		return 0.709
	default:
		return 0.7213 / (1 + 1.079/float64(m))
	}
}

// Estimate returns the approximate number of distinct items.
func (h *HLL) Estimate() uint64 {
	m := float64(h.m)
	var sum float64
	var zeros uint32
	if h.dense != nil {
		for _, r := range h.dense {
			sum += math.Ldexp(1, -int(r))
			if r == 0 {
				zeros++
			}
		}
	} else {
		present := uint32(len(h.sparse))
		zeros = h.m - present
		sum = float64(zeros) // each empty register contributes 2^0 = 1
		for _, r := range h.sparse {
			sum += math.Ldexp(1, -int(r))
		}
	}
	est := alphaM(h.m) * m * m / sum
	if est <= 2.5*m && zeros > 0 {
		est = m * math.Log(m/float64(zeros)) // linear counting
	}
	return uint64(est + 0.5)
}

// Merge folds other into h. Both must have the same precision; a mismatch returns
// an error wrapping ErrIncompatibleMerge.
func (h *HLL) Merge(other *HLL) error {
	if other.p != h.p {
		return fmt.Errorf("%w: HLL precisions %d and %d differ", ErrIncompatibleMerge, h.p, other.p)
	}
	if h.dense == nil && other.dense == nil && len(h.sparse)+len(other.sparse) <= h.sparseMax {
		for idx, r := range other.sparse {
			if cur, ok := h.sparse[idx]; !ok || r > cur {
				h.sparse[idx] = r
			}
		}
		return nil
	}
	if h.dense == nil {
		h.densify()
	}
	if other.dense != nil {
		for idx, r := range other.dense {
			if r > h.dense[idx] {
				h.dense[idx] = r
			}
		}
	} else {
		for idx, r := range other.sparse {
			if r > h.dense[idx] {
				h.dense[idx] = r
			}
		}
	}
	return nil
}

const (
	hllMagic0, hllMagic1, hllVersion = 'H', 'L', 1
	hllFlagSparse                    = 1
)

// Encode returns the binary form. A flag byte records whether the body is the
// sparse (index, rank) list or the dense register array.
func (h *HLL) Encode() []byte {
	var buf bytes.Buffer
	buf.WriteByte(hllMagic0)
	buf.WriteByte(hllMagic1)
	buf.WriteByte(hllVersion)
	if h.dense != nil {
		buf.WriteByte(0)
		buf.WriteByte(h.p)
		buf.Write(h.dense)
		return buf.Bytes()
	}
	buf.WriteByte(hllFlagSparse)
	buf.WriteByte(h.p)
	putUvarint(&buf, uint64(len(h.sparse)))
	// Encode in ascending index order so the output is deterministic.
	idxs := make([]uint32, 0, len(h.sparse))
	for idx := range h.sparse {
		idxs = append(idxs, idx)
	}
	slices.Sort(idxs)
	for _, idx := range idxs {
		putUvarint(&buf, uint64(idx))
		buf.WriteByte(h.sparse[idx])
	}
	return buf.Bytes()
}

// DecodeHLL parses the output of Encode. It validates the precision, bounds every
// index and rank by what the precision allows, and rejects a sparse count larger
// than m or than the remaining bytes can hold, so a malformed length cannot force a
// large allocation. It returns an error wrapping ErrBadData or ErrUnsupportedVersion.
func DecodeHLL(data []byte) (*HLL, error) {
	r := &reader{b: data}
	m0, err := r.readByte()
	if err != nil {
		return nil, err
	}
	m1, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if m0 != hllMagic0 || m1 != hllMagic1 {
		return nil, fmt.Errorf("%w: not an HLL sketch", ErrBadData)
	}
	ver, err := r.readByte()
	if err != nil {
		return nil, err
	}
	if ver != hllVersion {
		return nil, fmt.Errorf("%w: HLL version %d", ErrUnsupportedVersion, ver)
	}
	flags, err := r.readByte()
	if err != nil {
		return nil, err
	}
	pb, err := r.readByte()
	if err != nil {
		return nil, err
	}
	h, err := NewHLL(pb)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadData, err)
	}
	maxRank := uint8(65 - h.p)
	if flags&hllFlagSparse == 0 {
		if r.remaining() != int(h.m) {
			return nil, fmt.Errorf("%w: dense HLL needs %d register bytes, have %d", ErrBadData, h.m, r.remaining())
		}
		h.dense = make([]uint8, h.m)
		for i := uint32(0); i < h.m; i++ {
			b, _ := r.readByte()
			if b > maxRank {
				return nil, fmt.Errorf("%w: register rank %d over max %d", ErrBadData, b, maxRank)
			}
			h.dense[i] = b
		}
		h.sparse = nil
		return h, nil
	}
	count, err := r.readUvarint()
	if err != nil {
		return nil, err
	}
	if count > uint64(h.m) || count > uint64(r.remaining()) {
		return nil, fmt.Errorf("%w: sparse count %d over bound", ErrBadData, count)
	}
	for i := uint64(0); i < count; i++ {
		idx, err := r.readUvarint()
		if err != nil {
			return nil, err
		}
		rank, err := r.readByte()
		if err != nil {
			return nil, err
		}
		if idx >= uint64(h.m) {
			return nil, fmt.Errorf("%w: sparse index %d over m=%d", ErrBadData, idx, h.m)
		}
		if rank < 1 || rank > maxRank {
			return nil, fmt.Errorf("%w: sparse rank %d out of [1,%d]", ErrBadData, rank, maxRank)
		}
		if _, dup := h.sparse[uint32(idx)]; dup {
			return nil, fmt.Errorf("%w: duplicate sparse index %d", ErrBadData, idx)
		}
		h.sparse[uint32(idx)] = rank
	}
	if !r.atEnd() {
		return nil, fmt.Errorf("%w: %d trailing bytes", ErrBadData, r.remaining())
	}
	return h, nil
}
