We use Kaldi decoder to implement TLG based language model integration,
so we copied related files to this directory.
The main changes are:

1. To minimize the change, we use the same directories tree as Kaldi.

2. We replace Kaldi log system with glog in the following way.

``` c++
#define KALDI_WARN \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_WARNING).stream()
#define KALDI_ERR \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_ERROR).stream()
#define KALDI_INFO \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_INFO).stream()
#define KALDI_VLOG(v) VLOG(v)

#define KALDI_ASSERT(condition) CHECK(condition)
```

3. We lint all the files to satisfy the lint in WeNet.
