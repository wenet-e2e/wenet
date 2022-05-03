// Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef API_WENET_API_H_
#define API_WENET_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/** Init decoder from the file and returns the object
 *
 * @param model_dir: the model dir
 * @returns model object or NULL if problem occured
 */
void* wenet_init(const char* model_dir);


/** Free wenet decoder and corresponding resource
 */
void wenet_free(void* decoder);


/** Reset decoder for next decoding
 */
void wenet_reset(void* decoder);

/** Decode the input wav data
 * @param data: pcm data, encoded as int16_t(16 bits)
 * @param len: data length
 * @param last: if it is the last package
 */
void wenet_decode(void* decoder,
                  const char* data,
                  int len,
                  int last = 1);

/** Get decode result
 *  It returns partial result when finish is false
 *  It returns final result when finish is true
 */
const char* wenet_get_result(void* decoder);


/** Set log level
 *  We use glog in wenet, so the level is the glog level
 */
void wenet_set_log_level(int level);

#ifdef __cplusplus
}
#endif

#endif  // API_WENET_API_H_
