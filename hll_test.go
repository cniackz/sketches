package sketches

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func hllKey(i uint64) []byte {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], i)
	return b[:]
}

// TestHLLStandardError is sealed claim 3 (and claim 11, that splitmix64 is enough):
// over many independent trials the sample relative standard error is within a
// margin of 1.04/sqrt(m) and the mean bias is small, at precision 14.
func TestHLLStandardError(t *testing.T) {
	const p = 14
	m := float64(uint64(1) << p)
	theoRSE := 1.04 / math.Sqrt(m)
	for _, n := range []uint64{1000, 10000, 100000, 1000000} {
		trials := 2000000 / int(n)
		if trials < 30 {
			trials = 30
		}
		if trials > 200 {
			trials = 200
		}
		var sumErr, sumSq float64
		for tr := 0; tr < trials; tr++ {
			h, _ := NewHLL(p)
			base := uint64(tr+1) * 0x9e3779b97f4a7c15
			for i := uint64(0); i < n; i++ {
				h.Add(hllKey(base + i))
			}
			e := float64(h.Estimate())/float64(n) - 1
			sumErr += e
			sumSq += e * e
		}
		rse := math.Sqrt(sumSq / float64(trials))
		bias := sumErr / float64(trials)
		t.Logf("n=%d trials=%d: RSE=%.5f (theo %.5f), bias=%.5f", n, trials, rse, theoRSE, bias)
		if rse > 1.15*theoRSE {
			t.Fatalf("n=%d: sample RSE %.5f exceeds 1.15*theo %.5f", n, rse, 1.15*theoRSE)
		}
		if math.Abs(bias) > 0.01 {
			t.Fatalf("n=%d: bias %.5f exceeds 1%%", n, bias)
		}
	}
}

// TestHLLSmallRange is sealed claim 4: in the sparse, linear-counting regime the
// estimate is accurate.
func TestHLLSmallRange(t *testing.T) {
	const p = 14
	for _, n := range []uint64{10, 100, 1000, 3000} {
		var sumSq float64
		const trials = 100
		for tr := 0; tr < trials; tr++ {
			h, _ := NewHLL(p)
			base := uint64(tr+1) * 0xff51afd7ed558ccd
			for i := uint64(0); i < n; i++ {
				h.Add(hllKey(base + i))
			}
			e := float64(h.Estimate())/float64(n) - 1
			sumSq += e * e
		}
		rse := math.Sqrt(sumSq / trials)
		t.Logf("small n=%d: RSE=%.5f", n, rse)
		if rse > 0.03 {
			t.Fatalf("small n=%d: RSE %.5f too high for the linear-counting regime", n, rse)
		}
	}
}

func TestHLLMerge(t *testing.T) {
	const p = 14
	a, _ := NewHLL(p)
	b, _ := NewHLL(p)
	whole, _ := NewHLL(p)
	const n = 200000
	for i := uint64(0); i < n; i++ {
		if i%2 == 0 {
			a.Add(hllKey(i))
		} else {
			b.Add(hllKey(i))
		}
		whole.Add(hllKey(i))
	}
	if err := a.Merge(b); err != nil {
		t.Fatal(err)
	}
	em := float64(a.Estimate())
	ew := float64(whole.Estimate())
	if d := math.Abs(em-ew) / ew; d > 0.02 {
		t.Fatalf("merged estimate %.0f vs whole %.0f (rel %.4f)", em, ew, d)
	}
	bad, _ := NewHLL(12)
	if err := a.Merge(bad); err == nil {
		t.Fatal("merge of different precisions should fail")
	}
}

func TestHLLRoundTripBothFormats(t *testing.T) {
	const p = 14
	// Sparse: small cardinality stays sparse.
	sp, _ := NewHLL(p)
	for i := uint64(0); i < 500; i++ {
		sp.Add(hllKey(i))
	}
	if sp.dense != nil {
		t.Fatal("expected sparse at n=500")
	}
	// Dense: large cardinality densifies.
	dn, _ := NewHLL(p)
	for i := uint64(0); i < 100000; i++ {
		dn.Add(hllKey(i))
	}
	if dn.dense == nil {
		t.Fatal("expected dense at n=100000")
	}
	for _, h := range []*HLL{sp, dn} {
		enc := h.Encode()
		dec, err := DecodeHLL(enc)
		if err != nil {
			t.Fatal(err)
		}
		if dec.Estimate() != h.Estimate() {
			t.Fatalf("estimate differs after round trip: %d vs %d", dec.Estimate(), h.Estimate())
		}
		if !bytes.Equal(dec.Encode(), enc) {
			t.Fatal("re-encode differs")
		}
	}
}

func TestHLLParamValidation(t *testing.T) {
	for _, p := range []uint8{0, 1, 3, 19, 255} {
		if _, err := NewHLL(p); err == nil {
			t.Fatalf("NewHLL(%d) should fail", p)
		}
	}
}

// TestHLLTransitionBias is posterior H1: because the HLL++ bias-correction tables are
// not reproduced, the raw estimate is positively biased by a few percent in the
// transition band just above the linear-counting handoff (about 2.5*m to 3*m distinct
// items). This pins that documented limit so a regression cannot widen it silently:
// the band bias stays within 3%, and outside the band the bias returns to within 1%.
func TestHLLTransitionBias(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping bias sweep in short mode")
	}
	const p = 14
	m := uint64(1) << p
	meanBias := func(mult float64) float64 {
		n := uint64(mult * float64(m))
		const trials = 80
		var sum float64
		for tr := 0; tr < trials; tr++ {
			h, _ := NewHLL(p)
			base := uint64(tr+1) * 0x9e3779b97f4a7c15
			for i := uint64(0); i < n; i++ {
				h.Add(hllKey(base + i))
			}
			sum += float64(h.Estimate())/float64(n) - 1
		}
		return sum / trials
	}
	for _, mult := range []float64{2.4, 2.5, 2.7, 3.0} {
		b := meanBias(mult)
		t.Logf("transition n=%.1f*m: bias=%.5f", mult, b)
		if math.Abs(b) > 0.03 {
			t.Fatalf("transition bias %.5f at %.1f*m exceeds the documented 3%%", b, mult)
		}
	}
	for _, mult := range []float64{1.5, 2.0, 4.0, 6.0} {
		b := meanBias(mult)
		t.Logf("outside band n=%.1f*m: bias=%.5f", mult, b)
		if math.Abs(b) > 0.01 {
			t.Fatalf("bias %.5f at %.1f*m exceeds 1%% outside the transition band", b, mult)
		}
	}
}

func TestHLLDecodeRejectsGarbage(t *testing.T) {
	for _, bad := range [][]byte{
		{},
		{'X', 'Y', 1, 0, 14},
		{'H', 'L', 99, 0, 14},
		{'H', 'L', 1, 0, 99},             // bad precision
		{'H', 'L', 1, 0, 14},             // dense but no register bytes
		{'H', 'L', 1, 1, 14, 0xff, 0xff}, // sparse count huge
	} {
		if _, err := DecodeHLL(bad); err == nil {
			t.Fatalf("decode of %v should fail", bad)
		}
	}
}
