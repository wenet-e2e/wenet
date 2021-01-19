// Copyright 2011 Google Inc. All Rights Reserved.
// Author: settinger@google.com (Scott Ettinger)
// Simplified Google3 style logging with Android support.
// Supported macros are : LOG(INFO), LOG(WARNING), LOG(ERROR), LOG(FATAL),
//                        and VLOG(n).
//
// Portions of this code are taken from the GLOG package.  This code
// is only a small subset of the GLOG functionality. And like GLOG,
// higher levels are more verbose.
//
// Notable differences from GLOG :
//
// 1. lack of support for displaying unprintable characters and lack
// of stack trace information upon failure of the CHECK macros.
// 2. All output is tagged with the string "native".
// 3. While there is no runtime flag filtering logs (-v, -vmodule), the
//    compile time define MAX_LOG_LEVEL can be used to silence any
//    logging above the given level.
//
// -------------------------------- Usage ------------------------------------
// Basic usage :
// LOG(<severity level>) acts as a c++ stream to the Android logcat output.
// e.g. LOG(INFO) << "Value of counter = " << counter;
//
// Valid severity levels include INFO, WARNING, ERROR, FATAL.
// The various severity levels are routed to the corresponding Android logcat
// output.
// LOG(FATAL) outputs to the log and then terminates.
//
// VLOG(<severity level>) can also be used.
// VLOG(n) output is directed to the Android logcat levels as follows :
//  >=2 - Verbose
//    1 - Debug
//    0 - Info
//   -1 - Warning
//   -2 - Error
// <=-3 - Fatal
// Note that VLOG(FATAL) will terminate the program.
//
// CHECK macros are defined to test for conditions within code.  Any CHECK
// that fails will log the failure and terminate the application.
// e.g. CHECK_GE(3, 2) will pass while CHECK_GE(3, 4) will fail after logging
//      "Check failed 3 >= 4".
// The following CHECK macros are defined :
//
// CHECK(condition) - fails if condition is false and logs condition.
// CHECK_NOTNULL(variable) - fails if the variable is NULL.
//
// The following binary check macros are also defined :
//    Macro                 operator applied
// ------------------------------------------
// CHECK_EQ(val1, val2)      val1 == val2
// CHECK_NE(val1, val2)      val1 != val2
// CHECK_GT(val1, val2)      val1 > val2
// CHECK_GE(val1, val2)      val1 >= val2
// CHECK_LT(val1, val2)      val1 < val2
// CHECK_LE(val1, val2)      val1 <= val2
//
// Debug only versions of all of the check macros are also defined.  These
// macros generate no code in a release build, but avoid unused variable
// warnings / errors.
// To use the debug only versions, Prepend a D to the normal check macros.
// e.g. DCHECK_EQ(a, b);
#ifndef GLOG_LOGGING_H_
#define GLOG_LOGGING_H_
// Definitions for building on an Android system.
#include <android/log.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>
// Log severity level constants.
const int FATAL = -3;
const int ERROR = -2;
const int WARNING = -1;
const int INFO = 0;
// ------------------------- Glog compatibility ------------------------------
namespace google {
typedef int LogSeverity;
const int INFO = ::INFO;
const int WARNING = ::WARNING;
const int ERROR = ::ERROR;
const int FATAL = ::FATAL;
#ifdef ENABLE_LOG_SINKS
// Sink class used for integration with mock and test functions.
// If sinks are added, all log output is also sent to each sink through
// the send function.  In this implementation, WaitTillSent() is called
// immediately after the send.
// This implementation is not thread safe.
class LogSink {
 public:
  virtual ~LogSink() {}
  virtual void send(LogSeverity severity, const char* full_filename,
                    const char* base_filename, int line,
                    const struct tm* tm_time,
                    const char* message, size_t message_len) = 0;
  virtual void WaitTillSent() = 0;
};
// Global set of log sinks.
// TODO(settinger): Move this into a .cc file.
static std::set<LogSink *> log_sinks_global;
// Note: the Log sink functions are not thread safe.
inline void AddLogSink(LogSink *sink) {
  // TODO(settinger): Add locks for thread safety.
  log_sinks_global.insert(sink);
}
inline void RemoveLogSink(LogSink *sink) {
  log_sinks_global.erase(sink);
}
#endif  // #ifdef ENABLE_LOG_SINKS
inline void InitGoogleLogging(char *argv) {}
}  // namespace google
// ---------------------------- Logger Class --------------------------------
// Class created for each use of the logging macros.
// The logger acts as a stream and routes the final stream contents to the
// Android logcat output at the proper filter level.  This class should not
// be directly instantiated in code, rather it should be invoked through the
// use of the log macros LOG, or VLOG.
class MessageLogger {
 public:
  MessageLogger(const char *file, int line, const char *tag, int severity)
      : file_(file), line_(line), tag_(tag), severity_(severity) {
    // Pre-pend the stream with the file and line number.
    StripBasename(std::string(file), &filename_only_);
    stream_ << filename_only_ << ":" << line << " ";
  }
  // Output the contents of the stream to the proper channel on destruction.
  ~MessageLogger() {
#ifdef MAX_LOG_LEVEL
    if (severity_ > MAX_LOG_LEVEL && severity_ > FATAL) {
      return;
    }
#endif
    stream_ << "\n";
    static const int android_log_levels[] = {
        ANDROID_LOG_FATAL,    // LOG(FATAL)
        ANDROID_LOG_ERROR,    // LOG(ERROR)
        ANDROID_LOG_WARN,     // LOG(WARNING)
        ANDROID_LOG_INFO,     // LOG(INFO), VLOG(0)
        ANDROID_LOG_DEBUG,    // VLOG(1)
        ANDROID_LOG_VERBOSE,  // VLOG(2) .. VLOG(N)
    };
    // Bound the logging level.
    const int kMaxVerboseLevel = 2;
    int android_level_index = std::min(std::max(FATAL, severity_),
                                       kMaxVerboseLevel) - FATAL;
    int android_log_level = android_log_levels[android_level_index];
    // Output the log string the Android log at the appropriate level.
    __android_log_write(android_log_level, tag_.c_str(), stream_.str().c_str());
    // Indicate termination if needed.
    if (severity_ == FATAL) {
      __android_log_write(ANDROID_LOG_FATAL,
                          tag_.c_str(),
                          "terminating.\n");
    }
#ifdef ENABLE_LOG_SINKS
    LogToSinks(severity_);
    WaitForSinks();
#endif  // #ifdef ENABLE_LOG_SINKS
    // Android logging at level FATAL does not terminate execution, so abort()
    // is still required to stop the program.
    if (severity_ == FATAL) {
      abort();
    }
  }
  // Return the stream associated with the logger object.
  std::stringstream &stream() { return stream_; }
 private:
#ifdef ENABLE_LOG_SINKS
  void LogToSinks(int severity) {
    time_t rawtime;
    struct tm * timeinfo;
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    std::set<google::LogSink *>::iterator iter;
    // Send the log message to all sinks.
    for (iter = google::log_sinks_global.begin();
         iter != google::log_sinks_global.end(); ++iter)
      (*iter)->send(severity, file_.c_str(), filename_only_.c_str(), line_,
                    timeinfo, stream_.str().c_str(), stream_.str().size());
  }
  void WaitForSinks() {
    // TODO(settinger): add locks for thread safety.
    std::set<google::LogSink *>::iterator iter;
    // Call WaitTillSent() for all sinks.
    for (iter = google::log_sinks_global.begin();
         iter != google::log_sinks_global.end(); ++iter)
      (*iter)->WaitTillSent();
  }
#endif // #ifdef ENABLE_LOG_SINKS
  void StripBasename(const std::string &full_path, std::string *filename) {
    // TODO(settinger): add support for OS with different path separators.
    const char kSeparator = '/';
    size_t pos = full_path.rfind(kSeparator);
    if (pos != std::string::npos)
      *filename = full_path.substr(pos + 1, std::string::npos);
    else
      *filename = full_path;
  }
  std::string file_;
  std::string filename_only_;
  int line_;
  std::string tag_;
  std::stringstream stream_;
  int severity_;
};
// ---------------------- Logging Macro definitions --------------------------
// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LoggerVoidify {
 public:
  LoggerVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream &s) {}
};
// Log only if condition is met.  Otherwise evaluates to void.
#define LOG_IF(severity, condition) \
  !(condition) ? (void) 0 : LoggerVoidify() & \
    MessageLogger((char *)__FILE__, __LINE__, "native", severity).stream()
// Log only if condition is NOT met.  Otherwise evaluates to void.
#define LOG_IF_FALSE(severity, condition) LOG_IF(severity, !(condition))
// LG is a convenient shortcut for LOG(INFO). Its use is in new
// google3 code is discouraged and the following shortcut exists for
// backward compatibility with existing code.
#ifdef MAX_LOG_LEVEL
#define LOG(n) LOG_IF(n, n <= MAX_LOG_LEVEL)
#define VLOG(n) LOG_IF(n, n <= MAX_LOG_LEVEL)
#define LG LOG_IF(INFO, INFO <= MAX_LOG_LEVEL)
#else
#define LOG(n) MessageLogger((char *)__FILE__, __LINE__, "native", n).stream()
#define VLOG(n) MessageLogger((char *)__FILE__, __LINE__, "native", n).stream()
#define LG MessageLogger((char *)__FILE__, __LINE__, "native", INFO).stream()
#endif
// Currently, VLOG is always on for levels below MAX_LOG_LEVEL.
#ifndef MAX_LOG_LEVEL
#define VLOG_IS_ON(x) (1)
#else
#define VLOG_IS_ON(x) (x <= MAX_LOG_LEVEL)
#endif
#ifndef NDEBUG
#define DLOG LOG
#else
#define DLOG(severity) true ? (void) 0 : LoggerVoidify() & \
    MessageLogger((char *)__FILE__, __LINE__, "native", severity).stream()
#endif
// ---------------------------- CHECK helpers --------------------------------
// Log a message and terminate.
template<class T>
void LogMessageFatal(const char *file, int line, const T &message) {
  MessageLogger((char *) __FILE__, __LINE__, "native", FATAL).stream()
      << message;
}
// ---------------------------- CHECK macros ---------------------------------
// Check for a given boolean condition.
#define CHECK(condition) LOG_IF_FALSE(FATAL, condition) \
        << "Check failed: " #condition " "
#ifndef NDEBUG
// Debug only version of CHECK
#define DCHECK(condition) LOG_IF_FALSE(FATAL, condition) \
        << "Check failed: " #condition " "
#else
// Optimized version - generates no code.
#define DCHECK(condition) if (false) LOG_IF_FALSE(FATAL, condition) \
        << "Check failed: " #condition " "
#endif  // NDEBUG
// ------------------------- CHECK_OP macros ---------------------------------
// Generic binary operator check macro. This should not be directly invoked,
// instead use the binary comparison macros defined below.
#define CHECK_OP(val1, val2, op) LOG_IF_FALSE(FATAL, (val1 op val2)) \
  << "Check failed: " #val1 " " #op " " #val2 " "
// Check_op macro definitions
#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#ifndef NDEBUG
// Debug only versions of CHECK_OP macros.
#define DCHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#else
// These versions generate no code in optimized mode.
#define DCHECK_EQ(val1, val2) if (false) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) if (false) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) if (false) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) if (false) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) if (false) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) if (false) CHECK_OP(val1, val2, >)
#endif  // NDEBUG
// ---------------------------CHECK_NOTNULL macros ---------------------------
// Helpers for CHECK_NOTNULL(). Two are necessary to support both raw pointers
// and smart pointers.
template<typename T>
T &CheckNotNullCommon(const char *file, int line, const char *names, T &t) {
  if (t == NULL) {
    LogMessageFatal(file, line, std::string(names));
  }
  return t;
}
template<typename T>
T *CheckNotNull(const char *file, int line, const char *names, T *t) {
  return CheckNotNullCommon(file, line, names, t);
}
template<typename T>
T &CheckNotNull(const char *file, int line, const char *names, T &t) {
  return CheckNotNullCommon(file, line, names, t);
}
// Check that a pointer is not null.
#define CHECK_NOTNULL(val) \
  CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))
#ifndef NDEBUG
// Debug only version of CHECK_NOTNULL
#define DCHECK_NOTNULL(val) \
  CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))
#else
// Optimized version - generates no code.
#define DCHECK_NOTNULL(val) if (false)\
  CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))
#endif  // NDEBUG
inline void PrintAndroid(const char *msg) {
  __android_log_write(ANDROID_LOG_VERBOSE, "native", msg);
}
#endif  // GLOG_LOGGING_H_