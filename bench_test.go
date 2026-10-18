package sketches

import (
	"math/rand"
	"testing"
)

func BenchmarkDDSketchAdd(b *testing.B) {
	s, _ := NewDDSketch(0.01)
	rng := rand.New(rand.NewSource(1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Add(rng.ExpFloat64())
	}
}

func BenchmarkDDSketchQuantile(b *testing.B) {
	s, _ := NewDDSketch(0.01)
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 100000; i++ {
		s.Add(rng.ExpFloat64())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Quantile(0.99)
	}
}

func BenchmarkHLLAdd(b *testing.B) {
	h, _ := NewHLL(14)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.AddHash(uint64(i) * 0x9e3779b97f4a7c15)
	}
}

func BenchmarkHLLEstimate(b *testing.B) {
	h, _ := NewHLL(14)
	for i := 0; i < 1000000; i++ {
		h.AddHash(uint64(i) * 0x9e3779b97f4a7c15)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Estimate()
	}
}

func BenchmarkCountMinAdd(b *testing.B) {
	c, _ := NewCountMin(0.001, 0.01)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.AddHash(uint64(i)*0x9e3779b97f4a7c15, 1)
	}
}

func BenchmarkTDigestAdd(b *testing.B) {
	td, _ := NewTDigest(100)
	rng := rand.New(rand.NewSource(1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		td.AddWeighted(rng.NormFloat64(), 1)
	}
}

// TestMemoryFootprint records sealed claim 13: the per-structure memory.
func TestMemoryFootprint(t *testing.T) {
	h, _ := NewHLL(14)
	for i := 0; i < 1000000; i++ {
		h.AddHash(uint64(i) * 0x9e3779b97f4a7c15)
	}
	if len(h.dense) != 16384 {
		t.Fatalf("dense HLL at p=14 should be 16384 bytes, got %d", len(h.dense))
	}
	c, _ := NewCountMin(0.001, 0.01)
	t.Logf("count-min(0.001,0.01): %d cells = %d counters", len(c.counts), c.Width()*c.Depth())
	rng := rand.New(rand.NewSource(1))
	s, _ := NewDDSketch(0.01)
	for i := 0; i < 1000000; i++ {
		s.Add(rng.ExpFloat64())
	}
	t.Logf("ddsketch p=0.01, 1e6 exp values: %d buckets", s.NumBuckets())
}
