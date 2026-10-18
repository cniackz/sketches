package sketches

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func key(i int) []byte {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(i))
	return b[:]
}

// TestCountMinNeverUnderestimates is sealed claim 5: the estimate is never below
// the true count, for every key, on several adversarial streams.
func TestCountMinNeverUnderestimates(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, dist := range []string{"uniform", "zipf", "duplicates"} {
		c, err := NewCountMin(0.001, 0.01)
		if err != nil {
			t.Fatal(err)
		}
		truth := map[int]uint64{}
		var zipf *rand.Zipf
		if dist == "zipf" {
			zipf = rand.NewZipf(rng, 1.2, 1, 1<<20)
		}
		for n := 0; n < 200000; n++ {
			var k int
			switch dist {
			case "uniform":
				k = rng.Intn(50000)
			case "zipf":
				k = int(zipf.Uint64())
			case "duplicates":
				k = rng.Intn(100)
			}
			c.Add(key(k), 1)
			truth[k]++
		}
		for k, f := range truth {
			if est := c.Estimate(key(k)); est < f {
				t.Fatalf("%s: key %d estimate %d < true %d", dist, k, est, f)
			}
		}
	}
}

// TestCountMinProbabilisticBound is sealed claim 6: the fraction of keys whose
// overestimate exceeds epsilon*N is at most delta.
func TestCountMinProbabilisticBound(t *testing.T) {
	const epsilon, delta = 0.001, 0.05
	rng := rand.New(rand.NewSource(7))
	c, err := NewCountMin(epsilon, delta)
	if err != nil {
		t.Fatal(err)
	}
	truth := map[int]uint64{}
	zipf := rand.NewZipf(rng, 1.1, 1, 1<<20)
	for n := 0; n < 500000; n++ {
		k := int(zipf.Uint64())
		c.Add(key(k), 1)
		truth[k]++
	}
	bound := epsilon * float64(c.Total())
	var over, tot int
	for k, f := range truth {
		tot++
		if float64(c.Estimate(key(k))-f) > bound {
			over++
		}
	}
	frac := float64(over) / float64(tot)
	t.Logf("w=%d d=%d, %d/%d keys over epsilon*N (%.4f), delta=%.3f", c.Width(), c.Depth(), over, tot, frac, delta)
	if frac > delta {
		t.Fatalf("fraction over bound %.4f exceeds delta %.3f", frac, delta)
	}
}

func TestCountMinMerge(t *testing.T) {
	a, _ := NewCountMin(0.001, 0.01)
	b, _ := NewCountMin(0.001, 0.01)
	whole, _ := NewCountMin(0.001, 0.01)
	rng := rand.New(rand.NewSource(3))
	for n := 0; n < 100000; n++ {
		k := rng.Intn(20000)
		if n%2 == 0 {
			a.Add(key(k), 1)
		} else {
			b.Add(key(k), 1)
		}
		whole.Add(key(k), 1)
	}
	if err := a.Merge(b); err != nil {
		t.Fatal(err)
	}
	if a.Total() != whole.Total() {
		t.Fatalf("merged total %d != whole %d", a.Total(), whole.Total())
	}
	for i := range a.counts {
		if a.counts[i] != whole.counts[i] {
			t.Fatalf("cell %d: merged %d != whole %d", i, a.counts[i], whole.counts[i])
		}
	}
	bad, _ := NewCountMin(0.01, 0.01)
	if err := a.Merge(bad); err == nil {
		t.Fatal("merge of different shapes should fail")
	}
}

func TestCountMinRoundTrip(t *testing.T) {
	c, _ := NewCountMin(0.002, 0.02)
	rng := rand.New(rand.NewSource(5))
	for n := 0; n < 50000; n++ {
		c.Add(key(rng.Intn(10000)), 1+uint64(rng.Intn(3)))
	}
	enc := c.Encode()
	dec, err := DecodeCountMin(enc)
	if err != nil {
		t.Fatal(err)
	}
	if dec.w != c.w || dec.d != c.d || dec.total != c.total {
		t.Fatalf("shape or total differ after round trip")
	}
	if !bytes.Equal(dec.Encode(), enc) {
		t.Fatal("re-encode differs")
	}
	for n := 0; n < 100; n++ {
		k := key(rng.Intn(10000))
		if dec.Estimate(k) != c.Estimate(k) {
			t.Fatal("estimate differs after round trip")
		}
	}
}

func TestCountMinParamValidation(t *testing.T) {
	for _, tc := range []struct{ e, d float64 }{
		{0, 0.1}, {1, 0.1}, {-0.1, 0.1}, {0.1, 0}, {0.1, 1}, {math.NaN(), 0.1}, {0.1, math.Inf(1)},
	} {
		if _, err := NewCountMin(tc.e, tc.d); err == nil {
			t.Fatalf("NewCountMin(%g,%g) should fail", tc.e, tc.d)
		}
	}
	if _, err := NewCountMin(1e-12, 1e-12); err == nil {
		t.Fatal("absurd precision should be rejected by the cap")
	}
}

// TestCountMinRowIndependence is posterior CM2: the double-hashing scheme h1 + r*h2
// made two keys collide in ALL d rows with probability about 1/w^2 instead of 1/w^d,
// which broke the 1-delta bound for small delta. With independent per-row hashing an
// all-row collision has probability (1/w)^d, so over a million keys essentially none
// collide with a fixed heavy key in every row.
func TestCountMinRowIndependence(t *testing.T) {
	c, _ := NewCountMin(0.1, 1e-6) // w=28, d=14; (1/28)^14 is about 7e-21
	heavy := c.cols(hashBytes([]byte("heavy")))
	hcol := make([]uint32, c.d)
	for r := uint32(0); r < c.d; r++ {
		hcol[r] = heavy(r)
	}
	allCollide := 0
	for i := 0; i < 1_000_000; i++ {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(i))
		col := c.cols(hashBytes(b[:]))
		all := true
		for r := uint32(0); r < c.d; r++ {
			if col(r) != hcol[r] {
				all = false
				break
			}
		}
		if all {
			allCollide++
		}
	}
	// Independent rows: expectation ~0 over a million keys. The old double hashing made
	// it ~1/w^2 per key, over a thousand here.
	if allCollide > 2 {
		t.Fatalf("%d keys collide with the heavy key in all %d rows; rows are not independent", allCollide, c.d)
	}
}

// TestCountMinOverflowSaturates is posterior CM1: a count large enough to overflow a
// cell used to wrap, making Estimate fall below the true count and break the one-sided
// guarantee. Adds now saturate, so the estimate never underestimates.
func TestCountMinOverflowSaturates(t *testing.T) {
	c, _ := NewCountMin(0.01, 0.01)
	k1 := uint64(12345)
	col0 := c.cols(k1)(0)
	var k2 uint64
	for k := uint64(1); ; k++ {
		if k != k1 && c.cols(k)(0) == col0 {
			k2 = k
			break
		}
	}
	c.AddHash(k1, 1<<63)
	c.AddHash(k2, 1<<63)
	if e := c.EstimateHash(k1); e < 1<<63 {
		t.Fatalf("Estimate %d is below the true 2^63 (a cell wrapped instead of saturating)", e)
	}
}

// TestCountMinDecodeRejectsRowSum is posterior CM3: decode did not check that each row
// sums to the stored total, so a forged cell slipped through. It now rejects such a
// sketch. It also covers Merge(nil) and the zero value, which must not panic.
func TestCountMinDecodeRejectsRowSum(t *testing.T) {
	c, _ := NewCountMin(0.5, 0.5)
	c.Add([]byte("a"), 3)
	enc := c.Encode()
	enc[len(enc)-1] = 100 // inflate the last cell so its row no longer sums to the total
	if _, err := DecodeCountMin(enc); err == nil {
		t.Fatal("decode should reject a row that does not sum to the total")
	}
	good, _ := NewCountMin(0.01, 0.01)
	if err := good.Merge(nil); err == nil {
		t.Fatal("Merge(nil) should return an error, not panic")
	}
	var z CountMin
	z.Add([]byte("x"), 1) // zero value must not panic
	if z.EstimateHash(0) != 0 {
		t.Fatal("zero-value estimate should be 0")
	}
}

// TestCountMinRejectsOldFormat is posterior R2-F: the per-row hash family changed but
// the format version did not, so a blob written by the old code (version 1, correlated
// double hashing) decoded under the new hash and underestimated. Bumping the version
// makes decode reject a version-1 blob instead of reading it from the wrong columns.
func TestCountMinRejectsOldFormat(t *testing.T) {
	c, _ := NewCountMin(0.1, 0.1)
	c.Add([]byte("x"), 5)
	enc := c.Encode()
	if enc[2] != cmVersion {
		t.Fatalf("version byte is %d, expected %d", enc[2], cmVersion)
	}
	old := append([]byte(nil), enc...)
	old[2] = 1
	if _, err := DecodeCountMin(old); err == nil {
		t.Fatal("a version-1 blob used a correlated hash family and must be rejected")
	}
}

// TestCountMinSaturatedRoundTrip is posterior R2-G: a count large enough to saturate a
// cell (and the total) used to fail its own round trip, because decode rejected the
// overflowing row. Decode now sums each row with the same saturating add as Add, so a
// saturated sketch is accepted and encode/decode stay consistent.
func TestCountMinSaturatedRoundTrip(t *testing.T) {
	c, _ := NewCountMin(0.01, 0.01)
	c.AddHash(111, math.MaxUint64)
	c.AddHash(222, 1000) // pushes the total past MaxUint64, so it saturates
	if c.Total() != math.MaxUint64 {
		t.Fatalf("total %d should have saturated at MaxUint64", c.Total())
	}
	enc := c.Encode()
	dec, err := DecodeCountMin(enc)
	if err != nil {
		t.Fatalf("a saturated sketch must round trip, got: %v", err)
	}
	if !bytes.Equal(dec.Encode(), enc) {
		t.Fatal("re-encode of a saturated sketch differs")
	}
}

func TestCountMinDecodeRejectsGarbage(t *testing.T) {
	for _, bad := range [][]byte{
		{},
		{'X', 'Y', 1},
		{'C', 'M', 99},
		append([]byte{'C', 'M', 1}, make([]byte, 8)...),
	} {
		if _, err := DecodeCountMin(bad); err == nil {
			t.Fatalf("decode of %v should fail", bad)
		}
	}
	// A valid header with epsilon/delta but a truncated counter area.
	c, _ := NewCountMin(0.1, 0.1)
	enc := c.Encode()
	if _, err := DecodeCountMin(enc[:len(enc)-1]); err == nil {
		t.Fatal("truncated counters should fail")
	}
	_ = fmt.Sprint
}
