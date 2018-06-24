/**
 *   Copyright 2018 Ashar <ashar786khan@gmail.com>
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#ifndef SLICER_HPP
#define SLICER_HPP

#include <initializer_list>
#include "tensors++/exceptions/tensor_operation.hpp"

namespace tensors {
namespace slicer {
struct Slicer {
  std::initializer_list<uint> start, stop;
  uint step;

  Slicer(std::initializer_list<uint> A, std::initializer_list<uint> B,
         uint x = 1)
      : step(x), start(A), stop(B){};

  void validate() {
    if (start.size() != stop.size())
      throw exceptions::bad_slice(
          "The start and stop indices must match in dimensions.");
    if (step == 0) throw exceptions::bad_slice("Step size should not be zero");
    for (int t = 0; t < start.size(); t++) {
      if (*(start.begin() + t) < 0 || *(stop.begin() + t) < 0)
        throw exceptions::bad_slice("Negative indices are not allowed");
      if (*(start.begin() + t) > *(stop.begin() + t))
        throw exceptions::bad_slice(
            "Cannot slice when start index is more than end index at any "
            "dimensions");
    }
  }
};
}  // namespace slicer
}  // namespace tensors

#endif