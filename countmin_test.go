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
