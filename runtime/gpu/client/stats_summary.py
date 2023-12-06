#!/usr/bin/env python3
# Copyright      2023  Nvidia              (authors: Yuekai Zhang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert triton staistic json file for better view.

python3 stats_summary.py

"""
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--stats_file",
        type=str,
        required=False,
        default="./stats.json",
        help="output of stats anaylasis",
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        required=False,
        default="./stats_summary.txt",
        help="output of stats summary",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with open(args.stats_file) as stats_f, open(args.summary_file,
                                                "w") as summary_f:
        stats = json.load(stats_f)
        model_stats = stats["model_stats"]
        for model_state in model_stats:
            if "last_inference" not in model_state:
                continue
            summary_f.write(f"model name is {model_state['name']} \n")
            model_inference_stats = model_state["inference_stats"]
            total_queue_time_s = (int(model_inference_stats["queue"]["ns"]) /
                                  1e9)
            total_infer_time_s = (
                int(model_inference_stats["compute_infer"]["ns"]) / 1e9)
            total_input_time_s = (
                int(model_inference_stats["compute_input"]["ns"]) / 1e9)
            total_output_time_s = (
                int(model_inference_stats["compute_output"]["ns"]) / 1e9)
            summary_f.write(
                f"queue {total_queue_time_s:<5.2f} s, infer {total_infer_time_s:<5.2f} s, input {total_input_time_s:<5.2f} s, output {total_output_time_s:<5.2f} s \n"  # noqa
            )
            model_batch_stats = model_state["batch_stats"]
            for batch in model_batch_stats:
                batch_size = int(batch["batch_size"])
                compute_input = batch["compute_input"]
                compute_output = batch["compute_output"]
                compute_infer = batch["compute_infer"]
                batch_count = int(compute_infer["count"])
                assert (compute_infer["count"] == compute_output["count"] ==
                        compute_input["count"])
                compute_infer_time_ms = int(compute_infer["ns"]) / 1e6
                compute_input_time_ms = int(compute_input["ns"]) / 1e6
                compute_output_time_ms = int(compute_output["ns"]) / 1e6
                summary_f.write(
                    f"Batch_size {batch_size:<2}, {batch_count:<5} times, infer {compute_infer_time_ms:<9.2f} ms, avg {compute_infer_time_ms/batch_count:.2f} ms, {compute_infer_time_ms/batch_count/batch_size:.2f} ms "  # noqa
                )
                summary_f.write(
                    f"input {compute_input_time_ms:<9.2f} ms, avg {compute_input_time_ms/batch_count:.2f} ms, "  # noqa
                )
                summary_f.write(
                    f"output {compute_output_time_ms:<9.2f} ms, avg {compute_output_time_ms/batch_count:.2f} ms \n"  # noqa
                )
