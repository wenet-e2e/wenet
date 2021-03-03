//#include "fst/fstlib.h"

#include "utils/flags.h"
#include "utils/log.h"

DEFINE_int32(num, 10, "");

int main(int argc, char* argv[]) {
  ParseCommandLineFlags(&argc, &argv, false);
  LOG(INFO) << "Hello";
  return 0;
}
