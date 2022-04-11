package wenet

// #cgo CFLAGS: -I .
// #cgo LDFLAGS: -L .   -lcruntime
// #include "wenet.h"
// #include <stdlib.h>
import "C"

import (
	"os"
	"path"
	"runtime"
	"unsafe"
)

func Load(modelDir string, opts ...ParamsOpts) (*SimpleAsrModelWrapper, error) {
	zip := path.Join(modelDir, "final.zip")
	dict := path.Join(modelDir, "words.txt")
	if _, err := os.Stat(zip); err != nil {
		return nil, err
	}
	if _, err := os.Stat(dict); err != nil {
		return nil, err
	}
	if len(opts) == 0 {
		ctcopts := WithCtc(0, 10, 10)
		// feature pipeline opts
		fpopts := WithFeaturePipeline(80, 16000)
		modelopts := WithModel(zip, dict, 1)
		decodeopts := WithDecodeOpts(-1, 0.5, 0.5, 0)

		p := NewParams(ctcopts, fpopts, modelopts, decodeopts)
		return New(p), nil
	} else {
		p := NewParams(opts...)
		return New(p), nil
	}

}

func New(params *Params) *SimpleAsrModelWrapper {
	var model *C.struct_model
	model = C.wenet_init(params.params)

	m := &SimpleAsrModelWrapper{model}
	runtime.SetFinalizer(m, free)
	return m

}
func free(model *SimpleAsrModelWrapper) {
	C.wenet_free(model.inst)
}

type Params struct {
	params *C.struct_cParams
}

func NewParams(opts ...ParamsOpts) *Params {
	p := &Params{
		C.wenet_params_init(),
	}

	for _, opt := range opts {
		opt(p)
	}
	runtime.SetFinalizer(p, func(p *Params) {
		C.wenet_params_free(p.params)
	})
	return p
}

type ParamsOpts func(*Params)

func WithCtc(blank int, firstBeamSize int, secondBeamSize int) ParamsOpts {
	return func(p *Params) {
		opts := p.params
		C.wenet_params_set_ctc_opts(
			opts,
			C.int(blank),
			C.int(firstBeamSize),
			C.int(secondBeamSize),
		)
	}
}

func WithtWfst(
	maxActive int,
	minActive int,
	beam int,
	latticeBeam float64,
	acousticScale float64,
	blankSkipThresh float64,
	nbest int) ParamsOpts {
	return func(p *Params) {
		opts := p.params
		C.wenet_params_set_wfst_opts(
			opts,
			C.int(maxActive),
			C.int(minActive),
			C.int(beam),
			C.double(latticeBeam),
			C.double(acousticScale),
			C.double(blankSkipThresh),
			C.int(nbest),
		)
	}
}

func WithModel(modelpath string, dictpath string, numThreads int) ParamsOpts {
	return func(p *Params) {
		opts := p.params
		// c model path
		cmp := C.CString(modelpath)
		// c dict path
		cdp := C.CString(dictpath)

		defer C.free(unsafe.Pointer(cmp))
		defer C.free(unsafe.Pointer(cdp))
		C.wenet_params_set_model_opts(
			opts,
			cmp,
			cdp,
			C.int(numThreads),
		)
	}
}

func WithFeaturePipeline(numBins int, sampleRate int) ParamsOpts {
	return func(p *Params) {
		opts := p.params
		// c model path
		C.wenet_params_set_feature_pipeline_opts(
			opts,
			C.int(numBins),
			C.int(sampleRate),
		)
	}
}

func WithDecodeOpts(
	chunkSize int,
	ctcWeight float64,
	rescoreWeight float64,
	reverseWeight float64) ParamsOpts {
	return func(p *Params) {
		opts := p.params
		C.wenet_params_set_decode_opts(
			opts,
			C.int(chunkSize),
			C.double(ctcWeight),
			C.double(rescoreWeight),
			C.double(reverseWeight),
		)
	}
}

type SimpleAsrModelWrapper struct {
	inst *C.struct_model
}

func (samw *SimpleAsrModelWrapper) Recognize(pcm []byte, nbest int) string {
	if len(pcm) == 0 {
		return ""
	}
	cBytes := C.CBytes(pcm)
	defer C.free(cBytes)
	res := C.wenet_recognize(
		samw.inst,
		(*C.char)(cBytes),
		C.int(len(pcm)/2),
		C.int(nbest),
	)
	defer C.free(unsafe.Pointer(res))
	return C.GoString(res)
}

type StreammingAsrDecoder struct {
	decoder *C.struct_streamming_decoder

	// caller get decoding result from chan
	Result <-chan string
}

func NewStreammingAsrDecoder(samwp *SimpleAsrModelWrapper, nbest int, continuous_decoding bool) *StreammingAsrDecoder {
	if samwp == nil {
		return nil
	}
	cd := 0
	if continuous_decoding {
		cd = 1
	}
	d := C.streamming_decoder_init(
		samwp.inst,
		C.int(nbest),
		C.int(cd),
	)
	free := func(decoder *StreammingAsrDecoder) {
		C.streamming_decoder_free(decoder.decoder)
	}
	decoder := &StreammingAsrDecoder{
		decoder: d,
	}
	runtime.SetFinalizer(decoder, free)

	go decoder.asyncAsrRes()
	return decoder
}

func (sad *StreammingAsrDecoder) asyncAsrRes() {
	resultChan := make(chan string)
	sad.Result = resultChan
	var curResCstr *C.char
	for {
		finish := int(C.streamming_decoder_get_instance_result(sad.decoder, &curResCstr))
		curRes := C.GoString(curResCstr)
		C.free(unsafe.Pointer(curResCstr))

		resultChan <- curRes
		if finish != 0 {
			close(resultChan)
			break
		}
	}

}

func (sad *StreammingAsrDecoder) AcceptWaveform(pcm []byte, final bool) {
	if len(pcm) == 0 {
		return
	}
	cBytes := C.CBytes(pcm)
	defer C.free(cBytes)
	cfinal := 0
	if final {
		cfinal = 1
	}
	C.streamming_decoder_accept_waveform(
		sad.decoder,
		(*C.char)(cBytes),
		C.int(len(pcm)/2),
		C.int(cfinal),
	)

}

func (sad *StreammingAsrDecoder) Reset(nbest int, continuous_decoding bool) {
	// to int
	continuous_decoding_int := 0
	if continuous_decoding {
		continuous_decoding_int = 1
	}
	C.streamming_decoder_reset(
		sad.decoder,
		C.int(nbest),
		C.int(continuous_decoding_int),
	)
	go sad.asyncAsrRes()
}

type LabelChecker struct {
	// runtime keep alive when in c code
	model *SimpleAsrModelWrapper

	checker *C.struct_label_checker

	ISPenalty  float32
	DelPenalty float32
}

func NewLabelChecker(samwp *SimpleAsrModelWrapper) *LabelChecker {
	if samwp == nil {
		return nil
	}
	checker := C.label_checker_init(
		samwp.inst,
	)
	free := func(lcker *LabelChecker) {
		C.label_checker_free(lcker.checker)
	}
	labekChecker := &LabelChecker{
		model:      samwp,
		ISPenalty:  3.0,
		DelPenalty: 4.0,

		checker: checker,
	}
	runtime.SetFinalizer(labekChecker, free)

	return labekChecker
}

func (lcker *LabelChecker) Check(pcm []byte, labels []string) string {
	if len(pcm) == 0 {
		return ""
	}
	cBytes := C.CBytes(pcm)
	defer C.free(cBytes)

	cStrArray := make([]*C.char, len(labels))
	for i := range cStrArray {
		cStrArray[i] = C.CString(labels[i])
		defer C.free(unsafe.Pointer(cStrArray[i]))
	}
	res := C.label_checker_check(
		lcker.checker,

		(*C.char)(cBytes),
		C.int(len(pcm)/2),
		(**C.char)(unsafe.Pointer(&cStrArray[0])),
		C.int(len(labels)),

		C.float(lcker.ISPenalty),
		C.float(lcker.DelPenalty),
	)
	runtime.KeepAlive(lcker.model)

	defer C.free(unsafe.Pointer(res))
	return C.GoString(res)
}
