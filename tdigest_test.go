package sketches

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// rankOf returns the fraction of sorted values at most v (the empirical rank).
func rankOf(sorted []float64, v float64) float64 {
	i := sort.SearchFloat64s(sorted, v)
	// midpoint of the run, as a fraction of n
	return (float64(i) + 0.5) / float64(len(sorted))
}

// TestTDigestMonotone is part of sealed claim 7: the estimate is non-decreasing in
// q, on several distributions.
func TestTDigestMonotone(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	for _, kind := range []string{"normal", "pareto", "duplicates"} {
		td, _ := NewTDigest(100)
		data := ddDataset(rng, map[string]string{"normal": "mixed", "pareto": "pareto", "duplicates": "duplicates"}[kind], 100000)
		if kind == "normal" {
			data = data[:0]
			for i := 0; i < 100000; i++ {
				data = append(data, rng.NormFloat64())
			}
		}
		for _, v := range data {
			td.AddWeighted(v, 1)
		}
		prev := math.Inf(-1)
		for q := 0.0; q <= 1.0; q += 0.001 {
			got, err := td.Quantile(q)
			if err != nil {
				t.Fatal(err)
			}
			if got < prev-1e-9 {
				t.Fatalf("%s: quantile not monotone at q=%.3f: %g < %g", kind, q, got, prev)
			}
			prev = got
		}
	}
}

// TestTDigestTailResolution is part of sealed claim 7, as refined by posterior 7b:
// t-digest's real tail property is finer resolution, not smaller realized error.
// The clusters near q=0 and q=1 span a smaller fraction of the quantile range than
// the clusters near the median, which is what the scale function promises. (The
// realized rank error on a smooth density is dominated by interpolation and is not
// smaller at the tails; see SEALED-PREDICTION.md claim 7b.) The overall rank error
// stays small everywhere, which this test also checks.
func TestTDigestTailResolution(t *testing.T) {
	rng := rand.New(rand.NewSource(202))
	data := make([]float64, 200000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	td, _ := NewTDigest(100)
	for _, v := range data {
		td.AddWeighted(v, 1)
	}
	td.compress()
	// q-width (weight fraction) of the cluster whose center is nearest target q.
	widthNear := func(target float64) float64 {
		var cum, best, bestD float64
		bestD = math.Inf(1)
		for _, c := range td.merged {
			centerQ := (cum + c.weight/2) / td.totalWeight
			if d := math.Abs(centerQ - target); d < bestD {
				bestD = d
				best = c.weight / td.totalWeight
			}
			cum += c.weight
		}
		return best
	}
	wLow := widthNear(0.01)
	wHigh := widthNear(0.99)
	wMid := widthNear(0.5)
	t.Logf("cluster q-width: low=%.5f high=%.5f mid=%.5f", wLow, wHigh, wMid)
	if wLow >= wMid || wHigh >= wMid {
		t.Fatalf("tail clusters not finer than the median: low=%.5f high=%.5f mid=%.5f", wLow, wHigh, wMid)
	}
	// Realized rank error stays small everywhere.
	sorted := append([]float64(nil), data...)
	sort.Float64s(sorted)
	for _, q := range []float64{0.001, 0.01, 0.5, 0.99, 0.999} {
		v, _ := td.Quantile(q)
		if e := math.Abs(rankOf(sorted, v) - q); e > 0.01 {
			t.Fatalf("rank error at q=%g is %.5f, too high", q, e)
		}
	}
}

// TestTDigestMergeAccuracy is part of sealed claim 7: merging parts gives about the
// same quantiles as digesting the whole.
func TestTDigestMergeAccuracy(t *testing.T) {
	rng := rand.New(rand.NewSource(303))
	data := make([]float64, 300000)
	for i := range data {
		data[i] = math.Exp(2 * rng.NormFloat64())
	}
	whole, _ := NewTDigest(200)
	a, _ := NewTDigest(200)
	b, _ := NewTDigest(200)
	c, _ := NewTDigest(200)
	for i, v := range data {
		whole.AddWeighted(v, 1)
		switch i % 3 {
		case 0:
			a.AddWeighted(v, 1)
		case 1:
			b.AddWeighted(v, 1)
		default:
			c.AddWeighted(v, 1)
		}
	}
	if err := a.Merge(b); err != nil {
		t.Fatal(err)
	}
	if err := a.Merge(c); err != nil {
		t.Fatal(err)
	}
	sorted := append([]float64(nil), data...)
	sort.Float64s(sorted)
	for _, q := range []float64{0.01, 0.1, 0.5, 0.9, 0.99} {
		gm, _ := a.Quantile(q)
		rank := rankOf(sorted, gm)
		if math.Abs(rank-q) > 0.01 {
			t.Fatalf("merged q=%g lands at rank %.4f", q, rank)
		}
	}
	bad, _ := NewTDigest(100)
	if err := a.Merge(bad); err == nil {
		t.Fatal("merge of different compression should fail")
	}
}

func TestTDigestRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(404))
	td, _ := NewTDigest(100)
	for i := 0; i < 50000; i++ {
		td.AddWeighted(rng.NormFloat64(), 1)
	}
	enc := td.Encode()
	dec, err := DecodeTDigest(enc)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dec.Encode(), enc) {
		t.Fatal("re-encode differs")
	}
	for _, q := range []float64{0.01, 0.5, 0.99} {
		g1, _ := td.Quantile(q)
		g2, _ := dec.Quantile(q)
		if math.Abs(g1-g2) > 1e-9 {
			t.Fatalf("q=%g differs: %g vs %g", q, g1, g2)
		}
	}
}

func TestTDigestInvalidInput(t *testing.T) {
	td, _ := NewTDigest(100)
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if err := td.Add(v); err == nil {
			t.Fatalf("Add(%g) should fail", v)
		}
	}
	if err := td.AddWeighted(1, 0); err == nil {
		t.Fatal("weight 0 should fail")
	}
	if _, err := td.Quantile(0.5); err == nil {
		t.Fatal("empty digest Quantile should fail")
	}
	td.Add(1)
	for _, q := range []float64{-0.1, 1.1, math.NaN()} {
		if _, err := td.Quantile(q); err == nil {
			t.Fatalf("Quantile(%g) should fail", q)
		}
	}
	for _, comp := range []float64{0, 1, -5, math.Inf(1), math.NaN()} {
		if _, err := NewTDigest(comp); err == nil {
			t.Fatalf("NewTDigest(%g) should fail", comp)
		}
	}
}

// TestTDigestWeightOverflow is posterior T1: weights that sum past the float64 range
// used to make the total weight +Inf, which poisoned the scale function into NaN
// quantiles with no error. Add and Merge now reject the overflow and the digest stays
// usable.
func TestTDigestWeightOverflow(t *testing.T) {
	td, _ := NewTDigest(100)
	if err := td.AddWeighted(1, 1e308); err != nil {
		t.Fatalf("first 1e308 weight should be accepted: %v", err)
	}
	if err := td.AddWeighted(2, 1e308); err == nil {
		t.Fatal("a second 1e308 weight should be rejected as overflow")
	}
	q, err := td.Quantile(0.5)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(q) || math.IsInf(q, 0) {
		t.Fatalf("Quantile returned %g after a rejected overflow", q)
	}
	a, _ := NewTDigest(100)
	b, _ := NewTDigest(100)
	a.AddWeighted(1, 1e308)
	b.AddWeighted(2, 1e308)
	if err := a.Merge(b); err == nil {
		t.Fatal("a merge that overflows the total weight should be rejected")
	}
}

// TestTDigestDecodeRejectsBadMinMax is posterior T2: a decoded digest with centroids
// but a non-finite min or max used to be accepted and made Quantile return Inf. Decode
// now requires finite min and max when centroids are present.
func TestTDigestDecodeRejectsBadMinMax(t *testing.T) {
	td, _ := NewTDigest(100)
	for i := 0; i < 100; i++ {
		td.Add(float64(i))
	}
	enc := td.Encode()
	// Layout: magic(2) + version(1) + compression(8) + min(8) + max(8) + ...
	minOff, maxOff := 3+8, 3+8+8
	for _, tc := range []struct {
		name string
		off  int
		val  float64
	}{
		{"max=+Inf", maxOff, math.Inf(1)},
		{"min=-Inf", minOff, math.Inf(-1)},
		{"max=NaN", maxOff, math.NaN()},
		{"min=NaN", minOff, math.NaN()},
	} {
		b := append([]byte(nil), enc...)
		binary.LittleEndian.PutUint64(b[tc.off:], math.Float64bits(tc.val))
		if _, err := DecodeTDigest(b); err == nil {
			t.Fatalf("%s: decode should reject a non-finite min/max with centroids present", tc.name)
		}
	}
}

func TestTDigestDecodeRejectsGarbage(t *testing.T) {
	for _, bad := range [][]byte{
		{},
		{'X', 'Y', 1},
		{'T', 'D', 99},
	} {
		if _, err := DecodeTDigest(bad); err == nil {
			t.Fatalf("decode of %v should fail", bad)
		}
	}
	td, _ := NewTDigest(100)
	for i := 0; i < 1000; i++ {
		td.Add(float64(i))
	}
	enc := td.Encode()
	if _, err := DecodeTDigest(enc[:len(enc)-1]); err == nil {
		t.Fatal("truncated should fail")
	}
}
