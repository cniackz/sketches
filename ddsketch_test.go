package sketches

import (
	"bytes"
	"math"
	"math/rand"
	"sort"
	"testing"
)

func ddDataset(rng *rand.Rand, kind string, n int) []float64 {
	xs := make([]float64, 0, n)
	switch kind {
	case "pareto":
		for i := 0; i < n; i++ {
			u := rng.Float64()
			xs = append(xs, 1.0/math.Pow(1-u, 1.0/1.5))
		}
	case "lognormal":
		for i := 0; i < n; i++ {
			xs = append(xs, math.Exp(3*rng.NormFloat64()))
		}
	case "duplicates":
		vals := []float64{0.5, 1, 1, 1, 2, 5, 5, 100}
		for i := 0; i < n; i++ {
			xs = append(xs, vals[rng.Intn(len(vals))])
		}
	case "mixed":
		for i := 0; i < n; i++ {
			switch rng.Intn(3) {
			case 0:
				xs = append(xs, 0)
			case 1:
				xs = append(xs, 100*rng.NormFloat64())
			default:
				xs = append(xs, -math.Exp(2*rng.NormFloat64()))
			}
		}
	}
	return xs
}

func exactQuantile(sorted []float64, q float64) float64 {
	rank := q * float64(len(sorted)-1)
	return sorted[int(rank+0.5)]
}

// checkBound asserts the DDSketch relative-error guarantee at a grid of quantiles.
func checkBound(t *testing.T, s *DDSketch, data []float64, alpha float64, tag string) {
	t.Helper()
	sorted := append([]float64(nil), data...)
	sort.Float64s(sorted)
	for _, q := range []float64{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999} {
		got, err := s.Quantile(q)
		if err != nil {
			t.Fatalf("%s q=%g: %v", tag, q, err)
		}
		rank := q * float64(len(sorted)-1)
		lo := sorted[int(math.Floor(rank))]
		hi := sorted[int(math.Ceil(rank))]
		ok := false
		for _, xq := range []float64{lo, hi} {
			if xq == 0 {
				if got == 0 {
					ok = true
				}
				continue
			}
			if math.Abs(got-xq) <= alpha*math.Abs(xq)+1e-9 {
				ok = true
			}
		}
		if !ok {
			t.Fatalf("%s q=%g: got %g, true in [%g,%g], alpha=%g", tag, q, got, lo, hi, alpha)
		}
	}
}

// TestDDSketchRelativeError is sealed claim 1.
func TestDDSketchRelativeError(t *testing.T) {
	const alpha = 0.01
	rng := rand.New(rand.NewSource(11))
	for _, kind := range []string{"pareto", "lognormal", "duplicates", "mixed"} {
		s, _ := NewDDSketch(alpha)
		data := ddDataset(rng, kind, 100000)
		for _, v := range data {
			if err := s.Add(v); err != nil {
				t.Fatal(err)
			}
		}
		checkBound(t, s, data, alpha, kind)
		t.Logf("%s: %d buckets for %d values", kind, s.NumBuckets(), s.Count())
	}
}

// TestDDSketchMergePreservesBound is sealed claim 2.
func TestDDSketchMergePreservesBound(t *testing.T) {
	const alpha = 0.01
	rng := rand.New(rand.NewSource(22))
	data := ddDataset(rng, "lognormal", 120000)
	a, _ := NewDDSketch(alpha)
	b, _ := NewDDSketch(alpha)
	for i, v := range data {
		if i%2 == 0 {
			a.Add(v)
		} else {
			b.Add(v)
		}
	}
	if err := a.Merge(b); err != nil {
		t.Fatal(err)
	}
	checkBound(t, a, data, alpha, "merged")
	if a.Count() != uint64(len(data)) {
		t.Fatalf("merged count %d != %d", a.Count(), len(data))
	}
	bad, _ := NewDDSketch(0.02)
	if err := a.Merge(bad); err == nil {
		t.Fatal("merge of different alpha should fail")
	}
}

// TestDDSketchBounded checks that the cap holds and the high quantiles keep the
// guarantee even though the lowest-magnitude buckets were collapsed.
func TestDDSketchBounded(t *testing.T) {
	const alpha = 0.01
	rng := rand.New(rand.NewSource(33))
	s, _ := NewDDSketchBounded(alpha, 200)
	data := ddDataset(rng, "pareto", 100000)
	for _, v := range data {
		s.Add(v)
	}
	if len(s.pos) > 200 {
		t.Fatalf("positive buckets %d over cap 200", len(s.pos))
	}
	// The cap collapses the lowest-magnitude buckets, so the upper tail (the large
	// magnitudes these sketches are usually sized for) keeps the guarantee while the
	// smallest magnitudes do not. The far tail holds; the median of a heavy tail is
	// a small magnitude and may be collapsed, as documented.
	sorted := append([]float64(nil), data...)
	sort.Float64s(sorted)
	got, _ := s.Quantile(0.999)
	xq := exactQuantile(sorted, 0.999)
	if math.Abs(got-xq) > alpha*math.Abs(xq)+1e-9 {
		t.Fatalf("bounded tail q=0.999: got %g want ~%g", got, xq)
	}
	t.Logf("bounded pareto: %d pos buckets, p999 ok", len(s.pos))
}

func TestDDSketchRoundTrip(t *testing.T) {
	const alpha = 0.01
	rng := rand.New(rand.NewSource(44))
	s, _ := NewDDSketch(alpha)
	for _, v := range ddDataset(rng, "mixed", 50000) {
		s.Add(v)
	}
	enc := s.Encode()
	dec, err := DecodeDDSketch(enc)
	if err != nil {
		t.Fatal(err)
	}
	if dec.count != s.count || dec.zeroCount != s.zeroCount {
		t.Fatal("counts differ after round trip")
	}
	if !bytes.Equal(dec.Encode(), enc) {
		t.Fatal("re-encode differs")
	}
	for _, q := range []float64{0.01, 0.5, 0.99} {
		g1, _ := s.Quantile(q)
		g2, _ := dec.Quantile(q)
		if g1 != g2 {
			t.Fatalf("q=%g differs: %g vs %g", q, g1, g2)
		}
	}
}

func TestDDSketchInvalidInput(t *testing.T) {
	s, _ := NewDDSketch(0.01)
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if err := s.Add(v); err == nil {
			t.Fatalf("Add(%g) should fail", v)
		}
	}
	if s.Count() != 0 {
		t.Fatal("rejected adds should not change the sketch")
	}
	if _, err := s.Quantile(0.5); err == nil {
		t.Fatal("empty sketch Quantile should fail")
	}
	s.Add(1)
	for _, q := range []float64{-0.1, 1.1, math.NaN()} {
		if _, err := s.Quantile(q); err == nil {
			t.Fatalf("Quantile(%g) should fail", q)
		}
	}
	for _, a := range []float64{0, 1, -0.1, math.NaN()} {
		if _, err := NewDDSketch(a); err == nil {
			t.Fatalf("NewDDSketch(%g) should fail", a)
		}
	}
}

// TestDDSketchExtremeMagnitude is posterior D1: a magnitude so large that its bucket
// boundary overflows float64 used to make Quantile return +Inf. Add now rejects it
// (the guarantee cannot hold there), and a value just inside the range is accepted
// and returns a finite quantile.
func TestDDSketchExtremeMagnitude(t *testing.T) {
	s, _ := NewDDSketch(0.01)
	for _, v := range []float64{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64} {
		if err := s.Add(v); err == nil {
			t.Fatalf("Add(%g) should be rejected as out of representable range", v)
		}
	}
	if s.Count() != 0 {
		t.Fatal("rejected extreme adds should not change the sketch")
	}
	// A large but representable magnitude is accepted and its quantile stays finite.
	big := 1e300
	if err := s.Add(big); err != nil {
		t.Fatalf("Add(%g) should be accepted: %v", big, err)
	}
	got, err := s.Quantile(0.5)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(got, 0) || math.IsNaN(got) {
		t.Fatalf("Quantile returned %g for a representable value", got)
	}
	if math.Abs(got-big) > 0.01*big {
		t.Fatalf("Quantile %g not within alpha of %g", got, big)
	}
	// A crafted encoding with an out-of-range bucket index is rejected by decode.
	s2, _ := NewDDSketch(0.01)
	s2.Add(1)
	enc := s2.Encode()
	bad, err := DecodeDDSketch(enc)
	if err != nil {
		t.Fatal(err)
	}
	bad.pos[s2.maxIdx+1000] = 1
	bad.count++
	if _, err := DecodeDDSketch(bad.Encode()); err == nil {
		t.Fatal("decode of a non-representable bucket index should fail")
	}
}

// TestDDSketchAlphaFloor is posterior D2: an alpha below the float64 precision floor
// breaks the relative-error guarantee, so the constructor rejects it.
func TestDDSketchAlphaFloor(t *testing.T) {
	for _, a := range []float64{1e-15, 1e-12, 1e-10} {
		if _, err := NewDDSketch(a); err == nil {
			t.Fatalf("NewDDSketch(%g) should be rejected below the alpha floor", a)
		}
	}
	// The floor itself and anything above it is accepted.
	for _, a := range []float64{ddMinAlpha, 1e-6, 0.01} {
		if _, err := NewDDSketch(a); err != nil {
			t.Fatalf("NewDDSketch(%g) should be accepted: %v", a, err)
		}
	}
}

// TestDDSketchHugeCountQuantile is posterior D4: a (decoded) count above 2^53, where
// float64 cannot distinguish consecutive integers, used to panic in Quantile(1) when
// the positive store was empty. It must now return a finite value without panicking.
func TestDDSketchHugeCountQuantile(t *testing.T) {
	s, _ := NewDDSketch(0.01)
	// Only negatives, with a count above 2^53.
	s.neg[s.index(2)] = 1 << 53
	s.count = 1 << 53
	for _, q := range []float64{0, 0.5, 1} {
		v, err := s.Quantile(q)
		if err != nil {
			t.Fatalf("q=%g: %v", q, err)
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("q=%g returned %g", q, v)
		}
	}
	// The same through a decoded sketch, to exercise the decode path.
	dec, err := DecodeDDSketch(s.Encode())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dec.Quantile(1); err != nil {
		t.Fatalf("decoded Quantile(1): %v", err)
	}
}

func TestDDSketchDecodeRejectsGarbage(t *testing.T) {
	for _, bad := range [][]byte{
		{},
		{'X', 'Y', 1},
		{'D', 'D', 99},
	} {
		if _, err := DecodeDDSketch(bad); err == nil {
			t.Fatalf("decode of %v should fail", bad)
		}
	}
	s, _ := NewDDSketch(0.01)
	s.Add(1)
	s.Add(-2)
	enc := s.Encode()
	if _, err := DecodeDDSketch(enc[:len(enc)-1]); err == nil {
		t.Fatal("truncated should fail")
	}
}
