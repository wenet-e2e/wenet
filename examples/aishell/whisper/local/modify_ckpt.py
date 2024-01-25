# Copyright (c) 2023 Wenet Community. (authors: Xingchen Song)
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

import argparse
import torch
import json


def main():
    parser = argparse.ArgumentParser(description='filter out unused module')
    parser.add_argument('--remove_list',
                        default="name1,name2",
                        type=str,
                        help='list of name filter, comma-separated')
    parser.add_argument('--add_list',
                        default="{\"key1\": \"value1\"}",
                        type=str,
                        help='dict of name adder')
    parser.add_argument('--input_ckpt',
                        required=True,
                        type=str,
                        help='original checkpoint')
    parser.add_argument('--output_ckpt',
                        required=True,
                        type=str,
                        help='modified checkpoint')
    args = parser.parse_args()

    state = torch.load(args.input_ckpt, map_location="cpu")
    new_state = {}
    remove_list = args.remove_list.split(',')
    for k in state.keys():
        found = False
        for prefix in remove_list:
            if prefix in k:
                print("skip {}".format(k))
                found = True
                break
        if found:
            continue
        new_state[k] = state[k]
    add_list = json.loads(args.add_list)
    for k, v in add_list.items():
        print("init {} with {}".format(k, v))
        new_state[k] = state[v]
    torch.save(new_state, args.output_ckpt)


if __name__ == '__main__':
    main()
