#ifndef PYFLARE_CPP
#define PYFLARE_CPP

#include "Structure/py_structure.h"

PYBIND11_MODULE(_C_flare, m) {
  AddStructureModule(m);
}

#endif
