#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "runtime_wrapper.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_pywrap_wenet, m) {
  py::class_<Params>(m, "Params")
      .def(py::init<std::string, std::string, std::string, int, std::string,
                    std::string, double, int, int, int, int, double, double,
                    double, int, double, int, int, double, double, double, int,
                    int, double, double, int, bool>(),
           "model_path"_a = "", "dict_path"_a = "", "unit_path"_a = "",
           "num_threads"_a = 1, "fst_path"_a = "", "context_path"_a = "",
           "context_score"_a = 3.0, "num_bins"_a = 80, "sample_rate"_a = 16000,
           "chunk_size"_a = -1, "num_left_chunks"_a = -1, "ctc_weight"_a = 0.5,
           "rescoring_weight"_a = 0.5, "reverse_weight"_a = 0.0, "blank"_a = 0,
           "blank_threshold"_a = 0.8, "first_beam_size"_a = 10,
           "second_beam_size"_a = 10, "acoustic_scale"_a = 1.0,
           "nbest"_a = 10.0, "blank_skip_thresh"_a = 0.98,
           "max_active"_a = 7000, "min_active"_a = 200, "beam"_a = 16,
           "lattice_beam"_a = 16, "language_type"_a = 0, "lower_case"_a = true)
      .def_readwrite("model_path", &Params::model_path)
      .def_readwrite("dict_path", &Params::dict_path)
      .def_readwrite("unit_path", &Params::unit_path)
      .def_readwrite("num_threads", &Params::num_threads)
      .def_readwrite("fst_path", &Params::fst_path)
      .def_readwrite("context_path", &Params::context_path)
      .def_readwrite("context_score", &Params::context_score)
      .def_readwrite("num_bins", &Params::num_bins)
      .def_readwrite("sample_rate", &Params::sample_rate)
      .def_readwrite("chunk_size", &Params::chunk_size)
      .def_readwrite("num_left_chunks", &Params::num_left_chunks)
      .def_readwrite("ctc_weight", &Params::ctc_weight)
      .def_readwrite("rescoring_weight", &Params::rescoring_weight)
      .def_readwrite("reverse_weight", &Params::reverse_weight)
      .def_readwrite("blank", &Params::blank)
      .def_readwrite("blank_threshold", &Params::blank_threshold)
      .def_readwrite("first_beam_size", &Params::first_beam_size)
      .def_readwrite("second_beam_size", &Params::second_beam_size)
      .def_readwrite("acoustic_scale", &Params::acoustic_scale)
      .def_readwrite("nbest", &Params::nbest)
      .def_readwrite("blank_skip_thresh", &Params::blank_skip_thresh)
      .def_readwrite("max_active", &Params::max_active)
      .def_readwrite("min_active", &Params::min_active)
      .def_readwrite("beam", &Params::beam)
      .def_readwrite("lattice_beam", &Params::lattice_beam)
      .def_readwrite("language_type", &Params::language_type)
      .def_readwrite("lower_case", &Params::lower_case);

  py::class_<SimpleAsrModelWrapper, std::shared_ptr<SimpleAsrModelWrapper>>(
      m, "SimpleAsrModelWrapper")
      .def(py::init<const Params &>())
      .def("recognize", &SimpleAsrModelWrapper::Recognize);

  py::class_<StreammingAsrWrapper>(m, "StreammingAsrWrapper")
      .def(py::init<std::shared_ptr<SimpleAsrModelWrapper>, int, bool>())
      .def("AcceptWaveform",
           [](StreammingAsrWrapper &saw, char *pcm, int num_samples,
              bool final) {
             py::gil_scoped_release release;
             saw.AccepAcceptWaveform(pcm, num_samples, final);
           })
      .def("GetInstanceResult",
           [](StreammingAsrWrapper &saw) {
             py::gil_scoped_release release;
             std::string text;
             auto final = saw.GetInstanceResult(text);
             return std::make_tuple(text, final);
           })
      .def("Reset",
           [](StreammingAsrWrapper &saw, int nbest, bool continuous_decoding) {
             py::gil_scoped_release release;
             saw.Reset(nbest, continuous_decoding);
           });

  py::class_<LabelCheckerWrapper>(m, "LabelCheckerWrapper")
      .def(py::init<std::shared_ptr<SimpleAsrModelWrapper>>())
      .def("Check", &LabelCheckerWrapper::Check);
}
