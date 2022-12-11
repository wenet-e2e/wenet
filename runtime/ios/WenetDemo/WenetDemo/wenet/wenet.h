// Copyright (c) 2022 Dan Ma (1067837450@qq.com)
//
//  wenet.h
//  WenetDemo
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

#ifndef RUNTIME_IOS_WENETDEMO_WENETDEMO_WENET_WENET_H_
#define RUNTIME_IOS_WENETDEMO_WENETDEMO_WENET_WENET_H_

#include <stdio.h>

#import <Foundation/Foundation.h>

@interface Wenet : NSObject

- (nullable instancetype)initWithModelPath:
(NSString*)modelPath DictPath:(NSString*)dictPath;  // NOLINT

- (void)reset;

- (void)acceptWaveForm: (float*)pcm: (int)size;  // NOLINT

- (void)decode;

- (NSString*)get_result;  // NOLINT

@end

#endif  // RUNTIME_IOS_WENETDEMO_WENETDEMO_WENET_WENET_H_
