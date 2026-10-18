# sketches

Streaming summaries with error bounds, in pure Go with no dependencies beyond the
standard library:

- **DDSketch** quantiles with a proven relative-error guarantee.
- **t-digest** quantiles with strong tail accuracy at a small size (no formal bound).
- **HyperLogLog** cardinality with a known standard error.
- **count-min** frequencies with an additive, one-sided error bound.

Each summary is small, mergeable, and serializable. Every constructor validates its
parameters and returns an error instead of panicking, and no method panics on any
input. Decoders treat the bytes as untrusted and never allocate on a size the
parameters do not justify.

## Guarantees

| Structure | Guarantee | Parameter |
|---|---|---|
| DDSketch | `Quantile(q)` within `alpha` relative error of the true `x_q`, for every finite value whose magnitude is not at the extreme ends of the float64 range | relative accuracy `alpha` in `[1e-9, 1)` |
| HyperLogLog | relative standard error about `1.04/sqrt(m)`, `m = 2^p` (statistical, not per-estimate; see the transition-band note below) | precision `p` |
| count-min | `f(x) <= Estimate(x) <= f(x) + epsilon*N` with probability `>= 1-delta`, never under | `epsilon`, `delta` |
| t-digest | none in the worst case; empirically accurate, finest resolution in the tails | `compression` |

`DDSketch` rejects an `alpha` below `1e-9` (float64 cannot honor a finer relative
error) and rejects a value whose magnitude is so extreme that its bucket boundary
would overflow or underflow in float64. `HyperLogLog`, because it omits the HLL++
bias-correction tables, is positively biased by up to a few percent in a narrow
transition band just above where linear counting hands off to the raw estimate (about
`2.5*m` to `3*m` distinct items, measured about +2.4% at `p=14`); outside that band the
error returns to about `1.04/sqrt(m)`.

DDSketch carries a worst-case bound; t-digest does not. Use DDSketch when you need a
provable relative error, and t-digest when tail accuracy at a small size matters more.

## Usage

```go
dd, _ := sketches.NewDDSketch(0.01) // 1% relative accuracy
for _, v := range latencies {
    if err := dd.Add(v); err != nil { // rejects NaN and Inf
        // handle invalid input
    }
}
p99, _ := dd.Quantile(0.99)

hll, _ := sketches.NewHLL(14) // standard error about 0.81%
hll.Add([]byte("user-42"))
n := hll.Estimate()

cm, _ := sketches.NewCountMin(0.001, 0.01)
cm.Add([]byte("/api/x"), 1)
hits := cm.Estimate([]byte("/api/x"))

td, _ := sketches.NewTDigest(100)
td.Add(1.5)
median, _ := td.Quantile(0.5)
```

Every structure also has an `AddHash`/hash-free door where it takes an item, a
`Merge` for combining two summaries with the same parameters, and `Encode` with a
matching `Decode...` function.

## Merging and serialization

`Merge` combines two summaries built with the same parameters; a mismatch returns an
error wrapping `ErrIncompatibleMerge`. For DDSketch, HyperLogLog and count-min the
merge is exactly order-independent (it is bucket-wise addition or a register-wise
maximum). For t-digest the merge is only approximately order-independent: like adding
values in a different order, merging in a different order yields slightly different
centroids and so slightly different quantiles, within the usual empirical accuracy.
`Encode`
returns a versioned binary form that `DecodeDDSketch`, `DecodeHLL`,
`DecodeCountMin` and `DecodeTDigest` parse back. HyperLogLog has a sparse form for
low cardinality and a dense form for high; the format byte records which, and both
are covered by the decode fuzz tests. The count-min row hashes are derived
deterministically from the parameters (row `r` uses `h1 + r*h2`), so no seed is
stored and a decoded sketch uses the same hash family.

## Hashing

`Add([]byte)` uses a fixed internal mixer, the SplitMix64 finalizer, so results are
deterministic across processes and architectures and sketches merge anywhere.
Callers who already hold a 64-bit hash can use `AddHash` instead. The mixer is fixed
by the format version; changing it would change every HyperLogLog and count-min
result.

## Not in this version

Concurrency safety (synchronize externally), KLL and exponential-histogram quantile
sketches, sliding-window or time-decayed variants, arbitrary element types beyond
`[]byte` and `float64`, and the empirical bias-correction tables of HLL++ (linear
counting covers the small-cardinality regime instead).

## Performance

Measured on an arm64 laptop; your numbers will differ.

| Operation | Time | Allocations |
|---|---|---|
| DDSketch Add | ~26 ns | 0 |
| HyperLogLog Add | ~1.5 ns | 0 |
| count-min Add | ~5.5 ns | 0 |
| t-digest Add | ~97 ns | 0 amortized |

Memory: DDSketch grows with the logarithm of the data range (648 buckets for a
million exponential values at 1% accuracy), dense HyperLogLog is `2^p` bytes (16 KiB
at p=14), and count-min is `w*d` counters.

## License

Apache-2.0.
