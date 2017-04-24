#include "alchemist.h"
#include <cstdlib>

int main(int argc, char *argv[]) {
  using namespace alchemist;
  El::Initialize();
  El::Finalize();
  return EXIT_SUCCESS;
}
