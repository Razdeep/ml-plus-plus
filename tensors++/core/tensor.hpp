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

#include "tensors++/core/shape.hpp"
#include "tensors++/core/slicer.hpp"
#include "tensors++/core/tensor_config.hpp"
#include "tensors++/exceptions/tensor_formation.hpp"
#include "tensors++/exceptions/tensor_operation.hpp"

namespace tensors {

enum initializer { zeros, onces, random, uniform_gaussian, int_sequence };

typedef std::initializer_list<int> Indexer;

template <class dtype = float>
class tensor {
  shape::Shape shpe;
  size_t element_count;
  std::vector<uint> cum_shpe;  // cumulative shape dimension
  const config::Config &tensor_configuration;
  std::vector<dtype> data;
  initializer init_type;
  bool is_frozen = false;

  void init_initializer() {
    try {
      switch (init_type) {
        case zeros: {
          dtype T(0);  // this may throw if no constructor with int
          for (size_t i = 0; i < element_count; i++) data.push_back(T);
          break;
        }
        case onces: {
          dtype T(1);  // this may throw if no constructor with int
          for (size_t i = 0; i < element_count; i++) data.push_back(T);
          break;
        }
        case uniform_gaussian: {
          std::random_device rd;
          std::mt19937 gen(rd);
          std::normal_distribution<> d(0.0, 1.0);  // mean =0, varience =1
          for (size_t i = 0; i < element_count; i++) {
            dtype K(d(gen));  // this may throw if no constructor with float
            data.push_back(K);
          }
          break;
        }
        case random: {
          std::random_device rd;
          std::mt19937 gen(rd);
          std::uniform_real_distribution<> dist(0.0, 1.0);  // from [0,1)
          for (size_t i = 0; i < element_count; i++) {
            dtype K(dist(gen));  // this may throw if no constructor with float
            data.push_back(K);
          }
          break;
        }
        case int_sequence: {
          for (size_t i = 0; i < element_count; i++) {
            dtype K(static_cast<int>(i));
            data.push_back(K);
          }
          break;
        }
      };
    } catch (std::exception &e) {
      throw exceptions::initializer_exception(e.what());
    }
  }

  void update_shape(shape::Shape new_Shape) {
    shpe = new_Shape;
    cum_shpe = new_Shape.cumulative_shape();
    element_count = shpe.element_size();
  }

  size_t to_flat_index(Indexer &s) {
    if (s.size() != shpe.size())
      throw exceptions::bad_indexer(
          "Cannot flatten this Indexer has dimen " + std::to_string(s.size()) +
          "and Tensor has dimen " + std::to_string(shpe.size()));
    else {
      size_t ssf = 0;
      for (int t = 0; t < shpe.size(); t++) {
        if (*(s.begin() + t) > shpe[t])
          throw exceptions::bad_indexer(
              "Index out of range for dimension" + std::to_string(t) +
              "original tensor has shape index" +
              std::to_string(shpe[t] + ". Indexer has indexed " +
                             std::to_string(*(s.begin() + t))));
        ssf += *(s.begin() + t) * (element_count / cum_shpe[t]);
      }
      return ssf;
    }
  }

  void resize_shape(shape::Shape new_shape) {
    size_t old = element_count;
    size_t new_s = new_shape.element_size();
    if (new_s > old)
      for (size_t j = 0; j < (new_s - old); j++) data.push_back(dtype(0));
    if (new_s < old)
      for (size_t j = 0; j < (new_s - old); j++) data.pop_back();
    update_shape(new_shape);
  }

 public:
  tensor() = delete;

  // tensor: Parameterized Constructor
  tensor(
      shape::Shape shape,
      initializer init_method = initializer::uniform_gaussian,
      config::Config tensor_config = config::Config::default_config_instance())
      : tensor_configuration(tensor_config), init_type(init_method) {
    if (shape::Shape::is_initial_valid_shape(shape)) {
      update_shape(shape);
      init_initializer();
    } else
      throw exceptions::bad_init_shape(
          "Invalid Shape. All dimensions in the shape must be natural numbers "
          "(i.e > 0 )");
  }

  // tensor: Copy Constructor
  tensor(const tensor &ref) = default;

  // tensor: Move Constructor, does not throw any exception
  tensor(tensor &&that) noexcept {
    this->element_count = std::move(that.element_count);
    this->shpe = std::move(that.shpe);
    this->cum_shpe = std::move(that.cum_shape);
    this->is_frozen = std::move(that.is_frozen);
    this->init_type = std::move(that.init_type);
    this->tensor_configuration = std::move(that.tensor_configuration);
    this->data = std::move(that.data);
  }

  // inliners
  inline shape::Shape shape() const { return shpe; }
  inline size_t size() const { return element_count; }
  inline std::string data_type() const { return typeid(dtype).name(); }
  inline config::Config tensor_config() const { return tensor_configuration; }
  inline void unfreeze() { is_frozen = false; }

  // methods
  void freeze() {
    if (this->tensor_configuration.is_freezeable)
      is_frozen = true;
    else
      throw exceptions::operation_undefined(
          "Cannot Freeze a tensor that is declared unfreezable by its "
          "configuration.");
  }

  virtual tensor slice(slicer::Slicer &s) {}

  virtual bool reshape(std::initializer_list<int> &new_shape) {
    size_t ss = 1;
    int auto_shape = -1;
    std::vector<int> ns = new_shape;
    int running_index = 0;
    for (auto &e : new_shape) {
      if (e == 0)
        throw exceptions::bad_reshape(
            "New shape has an dimension with index ZERO.", 0, element_count);

      if (e < 0 && auto_shape) {
        throw exceptions::bad_reshape(
            "More than one dynamic size (-1) dimension found in reshape.", 0,
            element_count);
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
    if (ss == element_count)
      update_shape(shape::Shape(ns));
    else if (auto_shape != -1) {
      if (element_count % ss == 0) {
        uint dynamic_dimen = element_count / ss;
        ns[auto_shape] = dynamic_dimen;
        update_shape(shape::Shape(ns));
      } else
        throw exceptions::bad_reshape(
            "Cannot dynamically fit the data. Size axis mismatch",
            ss * (element_count / ss), element_count);
    } else
      throw exceptions::bad_reshape("Invalid reshape arguments", 0, 0);
  };

  virtual bool apply_lambda(std::function<void(dtype &)> op) final {
    for (int k = 0; k < element_count; k++) op(data[k]);
  }

  virtual tensor broadcast(const tensor &that) final {
    /// Returns the broadcasted version of this tensor w.r.t that
    // todo(coder3101) : Implement the broadcasting
    if (this->tensor_configuration.is_broad_castable) {
      //   uint lhs = this->shape().size();
      //   uint rhs = that.shape().size();
      //   if (lhs == rhs) {
      //     for (int t = lhs; t <= 0; t--) {
      //       if (this->shape()[t] != that.shape()[t] && this->shape()[t] != 1
      //       &&
      //           that.shape()[t] != 1)
      //         throw exceptions::broadcast_error(
      //             "Cannot broad-cast the two tensors.");
      //       if (this->shape()[t] == that.shape()[t]) }
      //   }
    } else
      throw exceptions::broadcast_error("Cannot broadcast" + this->shape() +
                                        " to : " + that.shape());
  };

  // all operations are element-wise and final
  virtual tensor operator+(const tensor &that) final {
    if (that.shape() != this->shape() &&
        !this->tensor_configuration.is_broad_castable &&
        !that.tensor_configuration.is_broad_castable) {
      throw exceptions::operation_undefined(
          "Element wise addition is not defined when both tensors are "
          "non-broadcastable and mismatch shape.");
    } else {
      if (shape::Shape::is_broadcastable_shape(*this, that)) {
        try {
          const tensor<dtype> &response = this->broadcast(that);
          tensor<dtype> res(response.shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = response.data[i] + that.data[i];
          return res;
        } catch (exceptions::broadcast_error &e) {
          const tensor<dtype> &response = that.broadcast(*this);
          tensor<dtype> res(this->shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = this->data[i] + response.data[i];
          return res;
        }
      } else
        throw exceptions::broadcast_error("Cannot broadcast tensor of shapes " +
                                          this->shape() + " and " +
                                          that.shape());
    }
  }
  virtual tensor<dtype> operator+(const dtype &k) final {
    tensor<dtype> result(this->shape());
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] + k;
    return result;
  }

  virtual tensor<dtype> operator++() final {
    for (size_t t = 0; t < element_count; t++) data[t]++;
    return *this;
  }
  virtual tensor<dtype> &operator--() final {
    for (size_t t = 0; t < element_count; t++) data[t]--;
    return *this;
  };

  virtual tensor operator-(const tensor &that) final {
    if (that.shape() != this->shape() &&
        !this->tensor_configuration.is_broad_castable &&
        !that.tensor_configuration.is_broad_castable) {
      throw exceptions::operation_undefined(
          "Element wise addition is not defined when both tensors are "
          "non-broadcastable and mismatch shape.");
    } else {
      if (shape::Shape::is_broadcastable_shape(*this, that)) {
        try {
          const tensor<dtype> &response = this->broadcast(that);
          tensor<dtype> res(response.shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = response.data[i] - that.data[i];
          return res;
        } catch (exceptions::broadcast_error &e) {
          const tensor<dtype> &response = that.broadcast(*this);
          tensor<dtype> res(this->shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = this->data[i] - response.data[i];
          return res;
        }
      } else
        throw exceptions::broadcast_error("Cannot broadcast tensor of shapes " +
                                          this->shape() + " and " +
                                          that.shape());
    }
  }
  virtual tensor<dtype> operator-(const dtype &k) final {
    tensor<dtype> result(this->shape());
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] - k;
    return result;
  }

  virtual tensor operator*(const tensor &that)final {
    if (that.shape() != this->shape() &&
        !this->tensor_configuration.is_broad_castable &&
        !that.tensor_configuration.is_broad_castable) {
      throw exceptions::operation_undefined(
          "Element wise addition is not defined when both tensors are "
          "non-broadcastable and mismatch shape.");
    } else {
      if (shape::Shape::is_broadcastable_shape(*this, that)) {
        try {
          const tensor<dtype> &response = this->broadcast(that);
          tensor<dtype> res(response.shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = response.data[i] * that.data[i];
          return res;
        } catch (exceptions::broadcast_error &e) {
          const tensor<dtype> &response = that.broadcast(*this);
          tensor<dtype> res(this->shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = this->data[i] * response.data[i];
          return res;
        }
      } else
        throw exceptions::broadcast_error("Cannot broadcast tensor of shapes " +
                                          this->shape() + " and " +
                                          that.shape());
    }
  };
  virtual tensor<dtype> operator*(const dtype &k)final {
    tensor<dtype> result(this->shape());
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] * k;
    return result;
  }

  virtual tensor operator/(const tensor &that) final {
    if (that.shape() != this->shape() &&
        !this->tensor_configuration.is_broad_castable &&
        !that.tensor_configuration.is_broad_castable) {
      throw exceptions::operation_undefined(
          "Element wise addition is not defined when both tensors are "
          "non-broadcastable and mismatch shape.");
    } else {
      if (shape::Shape::is_broadcastable_shape(*this, that)) {
        try {
          const tensor<dtype> &response = this->broadcast(that);
          tensor<dtype> res(response.shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = response.data[i] / that.data[i];
          return res;
        } catch (exceptions::broadcast_error &e) {
          const tensor<dtype> &response = that.broadcast(*this);
          tensor<dtype> res(this->shape());
          for (size_t i = 0; i < element_count; i++)
            res.data[i] = this->data[i] / response.data[i];
          return res;
        }
      } else
        throw exceptions::broadcast_error("Cannot broadcast tensor of shapes " +
                                          this->shape() + " and " +
                                          that.shape());
    }
  };
  virtual tensor<dtype> operator/(const dtype &k) final {
    tensor<dtype> result(this->shape());
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] / k;
    return result;
  }

  virtual bool operator==(const tensor &that) final {
    if (this->shape() != that.shape()) return false;
    for (size_t t = 0; t < element_count; t++)
      if (this->data[t] != that.data[t]) return false;

    return true;
  }
  virtual tensor &operator+=(const tensor &that) final;
  virtual tensor &operator-=(const tensor &that) final;
  virtual tensor &operator*=(const tensor &that) final;
  virtual tensor &operator+=(const dtype &t) final {
    for (size_t t = 0; t < element_count; t++) data[t] += t;
    return *this;
  }
  virtual tensor &operator-=(const dtype &t) final {
    for (size_t t = 0; t < element_count; t++) data[t] -= t;
    return *this;
  }
  virtual tensor &operator*=(const dtype &t) final {
    for (size_t t = 0; t < element_count; t++) data[t] *= t;
    return *this;
  }
  virtual dtype operator[](Indexer &p) final { return data[to_flat_index(p)]; };

  // methods
  virtual bool all(std::function<bool(dtype)>,
                   int axis = -1) final;  // True is all evalute to true
  virtual bool any(std::function<bool(dtype)>,
                   int axis = -1) final;  // True if any true
  virtual size_t argmax(int axis = -1) final;
  virtual size_t argmin(int axis = -1) final;
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