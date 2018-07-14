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
    if (s.size() != shpe.element_size())
      throw exceptions::bad_indexer(
          "Cannot flatten this Indexer has dimen " + std::to_string(s.size()) +
          "and Tensor has dimen " + std::to_string(shpe.element_size()));
    else {
      size_t ssf = 0;
      for (int t = 0; t < shpe.element_size(); t++) {
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

  tensor(
      std::vector<dtype> da, shape::Shape shape,
      config::Config tensor_config = config::Config::default_config_instance())
      : tensor_configuration(tensor_config) {
    if (shape::Shape::is_initial_valid_shape(shape)) {
      if (shape.element_size() == da.size()) {
        update_shape(shape);
        data = da;
      } else
        throw exceptions::bad_init_shape(
            "Invalid shape. The size of vector and shape do not match "
            "together.");
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
    if (this->tensor_configuration.is_freezeable) {
      is_frozen = true;
      this->data.shrink_to_fit();
    } else
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

  virtual std::vector<std::vector<dtype>> axis_wise(uint axis) final {
    std::vector<dtype> internal;
    std::vector<std::vector<dtype>> external;
    size_t repeat = shpe.reverse_cumulative_shape()[axis] / shpe[axis];
    size_t epoch = shpe.cumulative_shape()[axis] / shpe[axis];
    size_t current = shpe[axis];
    size_t last_repeat =
        axis == 0 ? 0
                  : shpe.reverse_cumulative_shape()[axis - 1] / shpe[axis - 1];

    for (size_t r = 0; r < repeat; r++) {
      for (size_t e = 0; e < epoch; e++) {
        for (size_t c = 0; c < current; c++) {
          internal.push_back(data[e * last_repeat + c * repeat + r]);
        }
        external.push_back(internal);
        internal.clear();
      }
      external.push_back(internal);
      internal.clear();
    }
    return external;
  }

  // all operations are element-wise and final
  virtual tensor operator+(const tensor &that) final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise addition is not defined when both tensors are mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      tensor<dtype> res(that.shape());
      for (size_t i = 0; i < element_count; i++)
        res.data[i] = this->data[i] + that.data[i];
    }
  }

  virtual tensor<dtype> operator+(const dtype &k) final {
    tensor<dtype> result(shpe);
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
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise subtraction is not defined when both tensors are "
          "mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      tensor<dtype> res(that.shape());
      for (size_t i = 0; i < element_count; i++)
        res.data[i] = this->data[i] - that.data[i];
    }
  }
  virtual tensor<dtype> operator-(const dtype &k) final {
    tensor<dtype> result(shpe);
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] - k;
    return result;
  }

  virtual tensor operator*(const tensor &that)final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise multiplication is not defined when both tensors are "
          "mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      tensor<dtype> res(that.shape());
      for (size_t i = 0; i < element_count; i++)
        res.data[i] = this->data[i] * that.data[i];
    }
  };
  virtual tensor<dtype> operator*(const dtype &k)final {
    tensor<dtype> result(shpe);
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] * k;
    return result;
  }

  virtual tensor operator/(const tensor &that) final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise division is not defined when both tensors are mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      tensor<dtype> res(that.shape());
      for (size_t i = 0; i < element_count; i++)
        res.data[i] = this->data[i] / that.data[i];
    }
  };
  virtual tensor<dtype> operator/(const dtype &k) final {
    tensor<dtype> result(shpe);
    for (size_t t = 0; t < element_count; t++)
      result.data[t] = this->data[t] / k;
    return result;
  }

  virtual bool operator==(const tensor &that) final {
    if (shpe != that.shape()) return false;
    for (size_t t = 0; t < element_count; t++)
      if (this->data[t] != that.data[t]) return false;

    return true;
  }
  virtual tensor &operator+=(const tensor &that) final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise addition is not defined when both tensors are mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      for (size_t i = 0; i < element_count; i++) this->data[i] += that.data[i];
      return *this;
    }
  };
  virtual tensor &operator-=(const tensor &that) final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise subtraction is not defined when both tensors are "
          "mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      for (size_t i = 0; i < element_count; i++) this->data[i] -= that.data[i];
      return *this;
    }
  };
  virtual tensor &operator*=(const tensor &that) final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise multiplication is not defined when both tensors are "
          "mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      for (size_t i = 0; i < element_count; i++) this->data[i] *= that.data[i];
      return *this;
    }
  };
  virtual tensor &operator/=(const tensor &that) final {
    if (that.shape() != shpe) {
      throw exceptions::operation_undefined(
          "Element wise division is not defined when both tensors are mismatch "
          "shape." +
          shpe + " and " + that.shape());
    } else {
      for (size_t i = 0; i < element_count; i++) this->data[i] /= that.data[i];
      return *this;
    }
  };
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
  virtual std::vector<dtype> operator[](const tensor<uint> &indexList) final {
    std::vector<dtype> res;
    if (indexList.shape().dimension() != 1)
      throw exceptions::operation_undefined(
          "Indexing tensor must be 1 dimensional");
    for (int k = 0; k < indexList.shape().element_size(); k++) {
      if (k > shpe.element_size())
        throw exceptions::operation_undefined(
            "Indexing tensor has value that is out of range for this tensor. "
            "Tried to access [" +
            std::to_string(k) + "] when max indexable is " +
            std::to_string(shpe.element_size()));
      res.push_back(indexList.data[k]);
    }
    return res;
  }
  // methods :: may change *this
  virtual bool all(std::function<bool(dtype)> op) final {
    for (size_t i = 0; i < element_count; i++)
      if (!op(this->data[i])) return false;
    return true;
  }
  virtual tensor<bool> all(std::function<bool(dtype)> op, int axis) final {
    if (shpe.dimension() <= axis)
      throw exceptions::axis_error(shpe.dimension() - 1, axis);
    else {
      std::vector<bool> res;
      std::vector<uint> ns;
      for (int t = 0; t < shpe.dimension(); t++)
        if (axis != t) ns.push_back(shpe[t]);
      std::vector<std::vector<dtype>> s = this->axis_wise(axis);
      bool flag_broken = false;
      for (auto &k : s) {
        for (int t = 0; t < s; t++) {
          if (!op(s[t])) {
            res.push_back(false);
            flag_broken = true;
            break;
          }
        }
        if (!flag_broken) res.push_back(true);
        flag_broken = false;
      }
      tensor<bool> r(res, shape::Shape(ns));
      return r;
    }
  }
  virtual bool any(std::function<bool(dtype)> op) final {
    for (size_t i = 0; i < element_count; i++)
      if (op(this->data[i])) return true;
    return false;
  }
  virtual tensor<bool> any(std::function<bool(dtype)> op, int axis) final {
    if (shpe.dimension() <= axis)
      throw exceptions::axis_error(shpe.dimension() - 1, axis);
    else {
      std::vector<bool> res;
      std::vector<uint> ns;
      for (int t = 0; t < shpe.dimension(); t++)
        if (axis != t) ns.push_back(shpe[t]);
      std::vector<std::vector<dtype>> s = this->axis_wise(axis);
      bool flag_broken = false;
      for (auto &k : s) {
        for (int t = 0; t < s; t++) {
          if (op(s[t])) {
            res.push_back(true);
            flag_broken = true;
            break;
          }
        }
        if (!flag_broken) res.push_back(false);
        flag_broken = false;
      }
      tensor<bool> r(res, shape::Shape(ns));
      return r;
    }
  }
  virtual void copy_to(tensor<dtype> &that,
                       bool explicitly_resize = false) final {
    if (!explicitly_resize && that.size() != this->size()) {
      throw exceptions::operation_undefined(
          "Cannot copy to target tensor this value. The sizes do not match and "
          "resize is set to false." +
          that.size() + " and " + this->size());
    } else {
      that.resize_shape(shpe);
      for (size_t t = 0; t < element_count; t++) that.data[t] = this->data[t];
    }
  }
  virtual size_t argmax(int axis = -1) final;
  virtual size_t argmin(int axis = -1) final;
  virtual void clip(dtype max, dtype min) final {
    for (auto &e : data) {
      if (e > max) e = max;
      if (e < min) e = min;
    }
  };
  virtual dtype cumulative_product(int axis = -1) final;
  virtual dtype cumulative_sum(int axis = -1) final;
  virtual tensor flatten() final;
  virtual dtype max(int axis = -1) final;
  virtual dtype min(int axis = -1) final;
  virtual dtype mean(int axis = -1) final;
  virtual dtype peek_to_peek(int axis = -1) final;  // max-min
  virtual void ravel() final { this->reshape(shape::Shape({element_count})); };
  virtual void swap_axis(int axis1, int axis2) final {
    if (axis1 >= shpe.dimension() || axis2 >= shpe.dimension())
      exceptions::operation_undefined(
          "Cannot swap axes. Range is out of bound for this tensor of "
          "dimensions" +
          std::to_string(shpe.dimension()));
    uint x = shpe[axis1];
    shpe[axis1] = shpe[axis2];
    shpe[axis2] = x;
    update_shape(shpe);
  };
  virtual void squeeze() final {
    std::vector<uint> newShape;
    for (auto &e : shpe)
      if (e != 1) newShape.push_back(e);
    update_shape(shape::Shape(newShape));
  };
  virtual dtype sum(int axis = -1) final;
  virtual dtype varience(int axis = -1) final;
};
}  // namespace tensors

#endif