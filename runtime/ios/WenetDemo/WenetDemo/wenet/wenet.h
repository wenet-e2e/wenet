//
//  wenet.h
//  WenetDemo

#ifndef wenet_h
#define wenet_h

#include <stdio.h>

#import <Foundation/Foundation.h>

@interface Wenet : NSObject

- (nullable instancetype)initWithModelPath:(NSString*)modelPath DictPath:(NSString*)dictPath;

- (void)reset;

- (void)acceptWaveForm: (float*)pcm: (int)size;

- (void)decode;

- (NSString*)get_result;

@end

#endif /* wenet_h */
