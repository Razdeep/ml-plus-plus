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

#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <initializer_list>
#include <string>
#include <vector>
#include <algorithm>

namespace tensors {
namespace shape {
struct Shape {
  std::vector<uint> d;

  Shape(std::initializer_list<uint> s) { d = s; }
  Shape(std::vector<uint> s) : d(s) {}
  Shape(std::vector<int> s) {
    for (auto &e : s) {
      if (e <= 0)
        d.push_back(0);
      else
        d.push_back(static_cast<uint>(e));
    }
  }
  Shape() = delete;

  size_t dimension() const { return d.size(); }

  std::vector<size_t> cumulative_shape() const {
    std::vector<size_t> res;
    size_t temp = 1;
    for (int t = 0; t < d.size(); t++) {
      temp *= d[t];
      res.push_back(temp);
    }
    return res;
  }

  std::vector<size_t> reverse_cumulative_shape() const {
    std::vector<size_t> res = cumulative_shape();
    std::reverse(std::begin(res), std::end(res));
    return res;
  }

  operator std::string() {
    std::string res = "(";
    for (auto &e : d) res += std::to_string(e) + ", ";
    *(res.end() - 2) = ')';
    res = res.substr(0, res.size() - 1);
    return res;
  }

  uint operator[](size_t a) { return d[a]; }
  uint operator==(const Shape &other) { return other.d == d; }

  size_t element_size() {
    size_t s = 1;
    for (auto &e : d) s *= e;
    return s;
  }

  static bool is_initial_valid_shape(const Shape &other) {
    for (auto &e : other.d)
      if (e <= 0) return false;
    return true;
  }
};

}  // namespace shape

}  // namespace tensors

#endif