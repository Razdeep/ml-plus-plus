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

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>
#include <initializer_list>
#include <memory>
#include <random>
#include <typeinfo>
#include <vector>

#include "tensors++/core/tensor_config.hpp"
#include "tensors++/exceptions/tensor_formation.hpp"
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
  };
};
}  // namespace slicer

enum initializer { zeros, onces, random, uniform_gaussian, int_sequence, none };

typedef std::initializer_list<uint> indexer;

typedef unsigned long long big_length;

template <class dtype = float>
class tensor {
  big_length sze;
  std::vector<int> shpe;
  bool is_frozen;
  bool was_dynamically_created;
  const config::Config &tensor_configuration;
  std::unique_ptr<dtype> data;
  initializer init_type;

  void init_memory() {
    if (sze > tensor_configuration.static_allocation_limits) {
      was_dynamically_created = true;
      data = new dtype[sze];
    } else {
      dtype s[sze];
      data = s[0];
      was_dynamically_created = false;
    }
  }

  void init_initializer() {
    try {
      switch (init_type) {
        case zeros: {
          dtype T(0);  // this may throw if no constructor with int
          for (big_length i = 0; i < sze; i++) *(data + i) = T;
          break;
        }
        case onces: {
          dtype T(1);  // this may throw if no constructor with int
          for (big_length i = 0; i < sze; i++) *(data + i) = T;
          break;
        }
        case uniform_gaussian: {
          std::random_device rd;
          std::mt19937 gen(rd);
          std::normal_distribution<> d(0.0, 1.0);  // mean =0, varience =1
          for (big_length i = 0; i < sze; i++) {
            dtype K(d(gen));  // this may throw if no constructor with float
            *(data + i) = K;
          }
          break;
        }
        case random: {
          std::random_device rd;
          std::mt19937 gen(rd);
          std::uniform_real_distribution<> dist(0.0, 1.0);  // from [0,1)
          for (big_length i = 0; i < sze; i++) {
            dtype K(dist(gen));  // this may throw if no constructor with float
            *(data + i) = K;
          }
          break;
        }
        case int_sequence: {
          for (big_length i = 0; i < sze; i++) {
            dtype K(static_cast<int>(i));
            *(data + i) = K;
          }
          break;
        }
        case none:
          // do nothing
          break;
      };
    } catch (std::exception &e) {
      throw exceptions::initializer_exception(e.what());
    }
  }

 public:
  tensor() = delete;

  // tensor: Parameterized Constructor
  tensor(
      std::initializer_list<int> shape,
      initializer init_method = initializer::uniform_gaussian,
      config::Config tensor_config = config::Config::default_config_instance())
      : tensor_configuration(tensor_config), init_type(init_method) {
    sze = 1;
    shpe = shape;
    for (auto &e : shape) {
      if (e <= 0)
        throw exceptions::bad_init_shape("A dimension has invalid size " +
                                         std::to_string(e));
      else
        sze *= e;
    }

    init_memory();
    init_initializer();
  }

  // tensor: Copy Constructor
  tensor(const tensor &ref) = default;

  // tensor: Move Constructor, does not throw any exception
  tensor(tensor &&that) noexcept {
    this->sze = std::move(that.sze);
    this->shpe = std::move(that.shpe);
    this->is_frozen = std::move(that.is_frozen);
    this->was_dynamically_created = std::move(that.was_dynamically_created);
    this->init_type = std::move(that.init_type);
    this->tensor_configuration = std::move(that.tensor_configuration);
    this->data = std::move(that.data);
  }
  std::vector<int> shape() const { return shpe; }

  unsigned long long size() const { return sze; }

  std::string data_type() const { return typeid(dtype).name(); }

  config::Config tensor_config() const { return tensor_configuration; }

  virtual void freeze() final {
    if (this->tensor_configuration.is_freezeable) {
      this->is_frozen = true;
    } else {
      throw exceptions::freeze_exception(
          "Change the config to make it freezeable.");
    }
  }

  virtual bool unfreeze() final {
    if (this->is_frozen) {
      this->is_frozen = false;
      return true;
    } else
      return false;
  }

  virtual tensor slice(slicer::Slicer &s) { s.validate(); }

  virtual bool reshape(std::initializer_list<int> &new_shape) {
    big_length ss = 1;
    int auto_shape = -1;
    int running_index = 0;
    for (auto &e : new_shape) {
      if (e == 0)
        throw exceptions::bad_reshape(
            "New shape has an dimension with index ZERO.", 0, sze);

      if (e < 0 && auto_shape) {
        throw exceptions::bad_reshape(
            "More than one dynamic size (-1) dimension found in reshape.", 0,
            sze);
      }

      if (e < 0) {
        auto_shape = true;
        auto_shape = running_index;
        running_index++;
        continue;
      }

      ss *= e;
      running_index++;
    }
    if (ss == sze)
      shpe = new_shape;
    else if (auto_shape != -1) {
      if (sze % ss == 0) {  // we can fit it dynamically
        uint dynamic_dimen = sze / ss;
        shpe = new_shape;
        *(shpe.begin() + auto_shape) = dynamic_dimen;
      } else
        throw exceptions::bad_reshape(
            "Cannot dynamically fit the data. Size axis mismatch",
            ss * (sze / ss), sze);
    } else
      throw exceptions::bad_reshape("Invalid reshape arguments", 0, 0);
  };

  virtual bool apply_lambda(std::function<void(dtype &)> op) final {
    for (int k = 0; k < sze; k++) op(data[k]);
  }

  virtual bool broadcast(uint axis) final {
    if (this->tensor_configuration.is_broad_castable) {
      // todo(coder3101) : Implement broadcast here
      return true;
    } else
      return false;  // was not able to broadcast
  }

  virtual bool broadcast(const tensor &that) final {
    if (this->tensor_configuration.is_broad_castable)
      // todo(coder3101) : Implement broadcast here
      return true;
    else
      return false;
  };

  // all operations are element-wise and final
  virtual tensor &operator+(tensor &that) final{};
  virtual tensor &operator++() final;
  virtual tensor &operator-(const tensor &that) final;
  virtual tensor &operator*(const tensor &that)final;
  virtual tensor &operator--() final;
  virtual bool operator==(const tensor &that) final;
  virtual tensor &operator+=(const tensor &that) final;
  virtual tensor &operator-=(const tensor &that) final;
  virtual tensor &operator*=(const tensor &that) final;
  virtual dtype operator[](indexer &p) final{};

  // methods
  virtual bool all(std::function<bool(dtype)>,
                   int axis = -1) final;  // True is all evalute to true
  virtual bool any(std::function<bool(dtype)>,
                   int axis = -1) final;  // True if any true
  virtual big_length argmax(int axis = -1) final;
  virtual big_length argmin(int axis = -1) final;
  virtual tensor clip(dtype max, dtype min) final;
  virtual tensor copy();
  virtual dtype cumulative_product(int axis = -1) final;
  virtual dtype cumulative_sum(int axis = -1) final;
  virtual tensor flatten() final;
  virtual dtype max(int axis = -1) final;
  virtual dtype min(int axis = -1) final;
  virtual dtype mean(int axis = -1) final;
  virtual dtype peek_to_peek(int axis = -1) final;  // max-min
  virtual void ravel() final;
  virtual void swap_axis(int axis1, int axis2) final;
  virtual void squeeze() final;
  virtual dtype sum(int axis = -1) final;
  virtual dtype varience(int axis = -1) final;
};
}  // namespace tensors

#endif