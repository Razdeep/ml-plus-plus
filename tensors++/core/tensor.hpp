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

#include "tensors++/core/slicer.hpp"
#include "tensors++/core/tensor_config.hpp"
#include "tensors++/exceptions/tensor_formation.hpp"
#include "tensors++/exceptions/tensor_operation.hpp"

namespace tensors {

enum initializer { zeros, onces, random, uniform_gaussian, int_sequence, none };

typedef unsigned long long big_num;

typedef std::initializer_list<uint> Indexer;

template <class dtype = float>
class tensor {
  big_num sze;
  std::vector<int> shpe;
  std::vector<uint> cum_shpe;  // cumulative shape dimension
  const config::Config &tensor_configuration;
  std::unique_ptr<dtype> data;
  initializer init_type;

  // private methods
  void init_memory() {
    if (sze > tensor_configuration.static_allocation_limits) {
      data = new dtype[sze];
    } else {
      dtype s[sze];
      data = s[0];
    }
  }
  void init_initializer() {
    try {
      switch (init_type) {
        case zeros: {
          dtype T(0);  // this may throw if no constructor with int
          for (big_num i = 0; i < sze; i++) *(data + i) = T;
          break;
        }
        case onces: {
          dtype T(1);  // this may throw if no constructor with int
          for (big_num i = 0; i < sze; i++) *(data + i) = T;
          break;
        }
        case uniform_gaussian: {
          std::random_device rd;
          std::mt19937 gen(rd);
          std::normal_distribution<> d(0.0, 1.0);  // mean =0, varience =1
          for (big_num i = 0; i < sze; i++) {
            dtype K(d(gen));  // this may throw if no constructor with float
            *(data + i) = K;
          }
          break;
        }
        case random: {
          std::random_device rd;
          std::mt19937 gen(rd);
          std::uniform_real_distribution<> dist(0.0, 1.0);  // from [0,1)
          for (big_num i = 0; i < sze; i++) {
            dtype K(dist(gen));  // this may throw if no constructor with float
            *(data + i) = K;
          }
          break;
        }
        case int_sequence: {
          for (big_num i = 0; i < sze; i++) {
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
  void update_shape(std::vector<int> new_Shape) {
    shpe = new_Shape;
    uint temp = new_Shape[0];
    cum_shpe.push_back(new_Shape[0]);
    for (int t = 1; t < new_Shape.size(); t++) {
      cum_shpe.push_back(temp * new_Shape[t]);
      temp *= new_Shape[t];
    }
  };
  big_num to_flat_index(Indexer &s) {
    if (s.size() != shpe.size())
      throw exceptions::bad_indexer(
          "Cannot flatten this Indexer has dimen " + std::to_string(s.size()) +
          "and Tensor has dimen " + std::to_string(shpe.size()));
    else {
      big_num ssf = 0;
      for (int t = 0; t < shpe.size(); t++)
        ssf += *(s.begin() + t) * cum_shpe[t];
      return ssf;
    }
  }
  void resize_memory(uint new_sze) {
    if (new_sze < tensor_configuration.static_allocation_limits &&
        sze < tensor_configuration.static_allocation_limits) {
      sze = new_sze;
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
    for (auto &e : shape) {
      if (e <= 0)
        throw exceptions::bad_init_shape("A dimension has invalid size " +
                                         std::to_string(e));
      else
        sze *= e;
    }

    update_shape(shape);
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
  big_num size() const { return sze; }
  std::string data_type() const { return typeid(dtype).name(); }
  config::Config tensor_config() const { return tensor_configuration; }

  virtual tensor slice(slicer::Slicer &s) { s.validate(); }

  virtual bool reshape(std::initializer_list<int> &new_shape) {
    big_num ss = 1;
    int auto_shape = -1;
    std::vector<int> ns = new_shape;
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
      update_shape(ns);
    else if (auto_shape != -1) {
      if (sze % ss == 0) {
        uint dynamic_dimen = sze / ss;
        ns[auto_shape] = dynamic_dimen;
        update_shape(ns);
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

  virtual bool broadcast(const tensor &that) final {
    // todo(coder3101) : Implement the broadcasting
    if (this->tensor_configuration.is_broad_castable) //{
    //   uint lhs = this->shape().size();
    //   uint rhs = that.shape().size();
    //   if (lhs == rhs) {
    //     for (int t = lhs; t <= 0; t--) {
    //       if (this->shape()[t] != that.shape()[t] && this->shape()[t] != 1 &&
    //           that.shape()[t] != 1)
    //         throw exceptions::broadcast_error(
    //             "Cannot broad-cast the two tensors.");
    //       if (this->shape()[t] == that.shape()[t]) }
    //   }
    // }
    return true;
    else return false;
  };

  // all operations are element-wise and final
  virtual tensor operator+(const tensor &that) final {}
  virtual tensor<dtype> operator+(const dtype &k) final {
    tensor<dtype> result(this->shape(), initializer::none);
    for (big_num t = 0; t < sze; t++) result.data[t] = this->data[t] + k;
    return result;
  }

  virtual tensor<dtype> operator++() final {
    for (big_num t = 0; t < sze; t++) data[t]++;
    return *this;
  }
  virtual tensor<dtype> &operator--() final {
    for (big_num t = 0; t < sze; t++) data[t]--;
    return *this;
  };

  virtual tensor operator-(const tensor &that) final;
  virtual tensor<dtype> operator-(const dtype &k) final {
    tensor<dtype> result(this->shape(), initializer::none);
    for (big_num t = 0; t < sze; t++) result.data[t] = this->data[t] - k;
    return result;
  }

  virtual tensor operator*(const tensor &that)final;
  virtual tensor<dtype> operator*(const dtype &k)final {
    tensor<dtype> result(this->shape(), initializer::none);
    for (big_num t = 0; t < sze; t++) result.data[t] = this->data[t] * k;
    return result;
  }

  virtual tensor operator/(const tensor &that) final;
  virtual tensor<dtype> operator/(const dtype &k) final {
    tensor<dtype> result(this->shape(), initializer::none);
    for (big_num t = 0; t < sze; t++) result.data[t] = this->data[t] / k;
    return result;
  }

  virtual bool operator==(const tensor &that) final {
    if (this->shape() != that.shape()) return false;
    for (big_num t = 0; t < sze; t++)
      if (this->data[t] != that.data[t]) return false;

    return true;
  }
  virtual tensor &operator+=(const tensor &that) final;
  virtual tensor &operator-=(const tensor &that) final;
  virtual tensor &operator*=(const tensor &that) final;
  virtual tensor &operator+=(const dtype &t) final {
    for (big_num t = 0; t < sze; t++) data[t] += t;
    return *this;
  }
  virtual tensor &operator-=(const dtype &t) final {
    for (big_num t = 0; t < sze; t++) data[t] -= t;
    return *this;
  }
  virtual tensor &operator*=(const dtype &t) final {
    for (big_num t = 0; t < sze; t++) data[t] *= t;
    return *this;
  }
  virtual dtype operator[](Indexer &p) final { return data[to_flat_index(p)]; };

  // methods
  virtual bool all(std::function<bool(dtype)>,
                   int axis = -1) final;  // True is all evalute to true
  virtual bool any(std::function<bool(dtype)>,
                   int axis = -1) final;  // True if any true
  virtual big_num argmax(int axis = -1) final;
  virtual big_num argmin(int axis = -1) final;
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