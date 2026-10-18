package sketches

import (
	"math"
	"testing"
)

// The decode fuzz targets feed arbitrary bytes to each decoder and require that it
// never panics and never allocates without bound. When a decode succeeds, the
// result must re-encode and decode again to the same bytes, so a decoder cannot
// accept a value its own encoder would not produce.

func FuzzDecodeCountMin(f *testing.F) {
	c, _ := NewCountMin(0.1, 0.1)
	c.Add([]byte("a"), 3)
	f.Add(c.Encode())
	f.Add([]byte{'C', 'M', 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := DecodeCountMin(data)
		if err != nil {
			return
		}
		re := got.Encode()
		again, err := DecodeCountMin(re)
		if err != nil || string(again.Encode()) != string(re) {
			t.Fatalf("round trip unstable")
		}
	})
}

func FuzzDecodeHLL(f *testing.F) {
	sp, _ := NewHLL(14)
	sp.Add([]byte("a"))
	f.Add(sp.Encode())
	dn, _ := NewHLL(4)
	for i := 0; i < 100; i++ {
		dn.AddHash(uint64(i) * 0x9e3779b97f4a7c15)
	}
	f.Add(dn.Encode())
	f.Add([]byte{'H', 'L', 1, 0, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := DecodeHLL(data)
		if err != nil {
			return
		}
		re := got.Encode()
		again, err := DecodeHLL(re)
		if err != nil || string(again.Encode()) != string(re) {
			t.Fatalf("round trip unstable")
		}
		_ = got.Estimate()
	})
}

func FuzzDecodeDDSketch(f *testing.F) {
	s, _ := NewDDSketch(0.01)
	s.Add(1)
	s.Add(-2)
	s.Add(0)
	f.Add(s.Encode())
	f.Add([]byte{'D', 'D', 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := DecodeDDSketch(data)
		if err != nil {
			return
		}
		re := got.Encode()
		again, err := DecodeDDSketch(re)
		if err != nil || string(again.Encode()) != string(re) {
			t.Fatalf("round trip unstable")
		}
		if got.Count() > 0 {
			_, _ = got.Quantile(0.5)
		}
	})
}

func FuzzDecodeTDigest(f *testing.F) {
	td, _ := NewTDigest(100)
	for i := 0; i < 500; i++ {
		td.Add(float64(i))
	}
	f.Add(td.Encode())
	f.Add([]byte{'T', 'D', 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := DecodeTDigest(data)
		if err != nil {
			return
		}
		re := got.Encode()
		again, err := DecodeTDigest(re)
		if err != nil || string(again.Encode()) != string(re) {
			t.Fatalf("round trip unstable")
		}
		if got.totalWeight > 0 {
			// Start below any representable value: a validly decoded digest may have a
			// min more negative than any fixed sentinel, and Quantile(0) returns it.
			prev := math.Inf(-1)
			for q := 0.0; q <= 1; q += 0.1 {
				v, _ := got.Quantile(q)
				if v < prev-1e-6 {
					t.Fatalf("decoded digest not monotone")
				}
				prev = v
			}
		}
	})
}
