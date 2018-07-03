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
#include <vector>
#include "tensors++/core/shape.hpp"
#include "tensors++/exceptions/tensor_operation.hpp"

namespace tensors {
namespace slicer {

#define END (-1)
#define BEGIN (-2)

struct Slicer {
  std::vector<uint> start, stop;
  const shape::Shape &original_shape;
  uint step;

  Slicer() = delete;

  Slicer(std::initializer_list<uint> A, std::initializer_list<uint> B,
         shape::Shape sp, uint x = 1)
      : step(x), start(A), stop(B), original_shape(sp) {
    validate();
  };

  Slicer(int full, std::initializer_list<uint> B, shape::Shape sp, uint x = 1)
      : step(x), stop(B), original_shape(sp) {
    if (full != BEGIN) {
      throw exceptions::bad_slice(
          "Starting location is undefined. If you wish to slice from begin "
          "first argument must be tensor::slice::BEGIN");
    } else {
      start = stop;
      for (auto &e : start) e = 0;
      validate();
    }
  }

  Slicer(std::initializer_list<uint> A, int full, shape::Shape sp, uint x = 1)
      : step(x), original_shape(sp), start(A) {
    if (full != END) {
      throw exceptions::bad_slice(
          "Ending location is undefined. Use tensor::slice::END");
    } else {
      stop = sp.d;
      validate();
    }
  }

  void validate() {
    if (start.size() != stop.size() ||
        start.size() != original_shape.dimension())
      throw exceptions::bad_slice(
          "The start and stop indices must match in dimensions.");
    if (step == 0) throw exceptions::bad_slice("Step size should not be zero");
    for (size_t t = 0; t < start.size(); t++) {
      if (start[t] > stop[t] || stop[t] > original_shape.d[t])
        throw exceptions::bad_slice(
            "Cannot slice when start index is more than end index at any "
            "dimensions or stop index is more than shape");
    }
  }
};
}  // namespace slicer
}  // namespace tensors

#endif