package sketches

import (
	"bytes"
	"encoding/binary"
	"math"
)

// The encoders write a small header (a two-byte magic and a one-byte version)
// followed by the parameters and then the data, all little-endian. The decoders
// never trust the input: every length is checked against both the remaining bytes
// and the bound the declared parameters allow before anything is allocated.

func putUvarint(buf *bytes.Buffer, x uint64) {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(tmp[:], x)
	buf.Write(tmp[:n])
}

// putZigzag encodes a signed integer so that small magnitudes stay small.
func putZigzag(buf *bytes.Buffer, x int64) {
	putUvarint(buf, uint64((x<<1)^(x>>63)))
}

func putFloat(buf *bytes.Buffer, f float64) {
	var tmp [8]byte
	binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(f))
	buf.Write(tmp[:])
}

// reader consumes an encoded buffer, returning ErrBadData on any truncation.
type reader struct {
	b []byte
	i int
}

func (r *reader) readByte() (byte, error) {
	if r.i >= len(r.b) {
		return 0, ErrBadData
	}
	v := r.b[r.i]
	r.i++
	return v, nil
}

func (r *reader) readUvarint() (uint64, error) {
	x, n := binary.Uvarint(r.b[r.i:])
	if n <= 0 {
		return 0, ErrBadData
	}
	r.i += n
	return x, nil
}

func (r *reader) readZigzag() (int64, error) {
	u, err := r.readUvarint()
	if err != nil {
		return 0, err
	}
	return int64(u>>1) ^ -int64(u&1), nil
}

func (r *reader) readFloat() (float64, error) {
	if r.i+8 > len(r.b) {
		return 0, ErrBadData
	}
	u := binary.LittleEndian.Uint64(r.b[r.i:])
	r.i += 8
	return math.Float64frombits(u), nil
}

// remaining is how many bytes are left to read, used to reject a declared count
// that cannot fit even at one byte each before any slice is allocated.
func (r *reader) remaining() int { return len(r.b) - r.i }

func (r *reader) atEnd() bool { return r.i >= len(r.b) }
