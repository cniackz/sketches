package sketches

import "encoding/binary"

// mix64 is the SplitMix64 finalizer (Vigna, public domain): a bijection on uint64
// with strong avalanche. It is the one mixing primitive this package uses, so the
// hash family is fixed and deterministic across processes and architectures, which
// is what lets an encoded sketch be decoded and merged anywhere.
func mix64(z uint64) uint64 {
	z += 0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// hashBytes maps a byte slice to a 64-bit value deterministically, folding each
// little-endian 8-byte word through the SplitMix64 finalizer with the length mixed
// in so that inputs of different length do not collide trivially. It is fixed by
// the format version: changing it would change every HyperLogLog and count-min
// result, so a new mixer would be a new format version.
func hashBytes(b []byte) uint64 {
	h := 0x9e3779b97f4a7c15 ^ uint64(len(b))
	i := 0
	for ; i+8 <= len(b); i += 8 {
		h = mix64(h ^ binary.LittleEndian.Uint64(b[i:]))
	}
	if i < len(b) {
		var tail uint64
		for j := 0; i+j < len(b); j++ {
			tail |= uint64(b[i+j]) << (8 * j)
		}
		h = mix64(h ^ tail)
	}
	return mix64(h)
}
