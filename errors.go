package sketches

import "errors"

// ErrInvalidParam is returned by a constructor when a parameter is out of range,
// such as an alpha outside (0,1) or a precision outside the supported range. Use
// errors.Is to test for it.
var ErrInvalidParam = errors.New("sketches: invalid parameter")

// ErrInvalidInput is returned by an estimator when an input is not usable, such as
// a NaN or infinite value added to a quantile sketch, or a quantile query outside
// [0,1]. Use errors.Is to test for it.
var ErrInvalidInput = errors.New("sketches: invalid input")

// ErrEmpty is returned by a query against a summary that has seen no data. Use
// errors.Is to test for it.
var ErrEmpty = errors.New("sketches: empty summary")

// ErrIncompatibleMerge is returned by Merge when the two summaries were built with
// different parameters and cannot be combined. Use errors.Is to test for it.
var ErrIncompatibleMerge = errors.New("sketches: incompatible summaries")

// ErrBadData is returned by a decoder when the encoded bytes are malformed: a bad
// magic, a truncated field, an out-of-range parameter, or a declared size the
// parameters cannot justify. Use errors.Is to test for it.
var ErrBadData = errors.New("sketches: malformed encoded data")

// ErrUnsupportedVersion is returned by a decoder when the format version byte is
// one this package does not know. Use errors.Is to test for it.
var ErrUnsupportedVersion = errors.New("sketches: unsupported format version")
