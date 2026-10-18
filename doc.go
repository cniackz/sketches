// Package sketches implements streaming summaries with error bounds: DDSketch
// and t-digest for quantiles, HyperLogLog for cardinality, and count-min for
// frequencies. Each summary is small, mergeable, and serializable. Where a
// structure carries a proven error guarantee the godoc states it in terms a test
// can check; t-digest, which has no worst-case guarantee, says so plainly.
//
// All constructors validate their parameters and return an error rather than
// panic, and no method panics on any input. The estimators reject a non-finite
// numeric input with an error wrapping ErrInvalidInput. Decoders treat every byte
// as untrusted: an unknown version, an out-of-range parameter, or a declared size
// that the parameters cannot justify is rejected with an error, never a panic or
// an unbounded allocation.
//
// The summaries are not safe for concurrent use; synchronize externally.
//
// References, used clean-room from the published papers: Masson, Rong and Nguyen
// 2019 (DDSketch); Dunning and Ertl (t-digest); Flajolet, Fusy, Gandon and
// Meunier 2007 and Heule, Nunkesser and Hall 2013 (HyperLogLog and HLL++);
// Cormode and Muthukrishnan 2005 (count-min).
package sketches
