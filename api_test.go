package sketches

import "testing"

// TestMergeNilAndZeroValue is posterior API: Merge(nil) used to panic on all four
// summaries, and a zero-value receiver (built without a constructor) panicked in Add.
// No method may panic on a constructed value, and the zero value must fail cleanly
// rather than crash. This runs each case and fails if any panics.
func TestMergeNilAndZeroValue(t *testing.T) {
	run := func(name string, f func()) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("%s panicked: %v", name, r)
			}
		}()
		f()
	}

	dd, _ := NewDDSketch(0.01)
	h, _ := NewHLL(14)
	cm, _ := NewCountMin(0.01, 0.01)
	td, _ := NewTDigest(100)

	run("DDSketch.Merge(nil)", func() {
		if err := dd.Merge(nil); err == nil {
			t.Error("DDSketch.Merge(nil) should return an error")
		}
	})
	run("HLL.Merge(nil)", func() {
		if err := h.Merge(nil); err == nil {
			t.Error("HLL.Merge(nil) should return an error")
		}
	})
	run("CountMin.Merge(nil)", func() {
		if err := cm.Merge(nil); err == nil {
			t.Error("CountMin.Merge(nil) should return an error")
		}
	})
	run("TDigest.Merge(nil)", func() {
		if err := td.Merge(nil); err == nil {
			t.Error("TDigest.Merge(nil) should return an error")
		}
	})

	run("DDSketch{}.Add", func() {
		var z DDSketch
		if err := z.Add(1); err == nil {
			t.Error("zero-value DDSketch.Add should return an error")
		}
	})
	run("HLL{}.Add", func() { var z HLL; z.Add([]byte("x")) })
	run("CountMin{}.Add", func() { var z CountMin; z.Add([]byte("x"), 1) })
	run("TDigest{}.Add+Quantile", func() {
		var z TDigest
		_ = z.Add(1)
		_, _ = z.Quantile(0.5)
	})
}
