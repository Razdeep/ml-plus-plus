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
#include <vector>

#include "tensors++/core/tensor_config.hpp"

namespace tensors {

template <class dtype>
class tensor {
  unsigned long long sze;
  std::vector<int> shpe;
  bool is_frozen;
  const config::Config &tensor_configuration;
  std::vector<std::vector<int>> axis_manager;
  std::unique_ptr<dtype *> data;

 public:
  std::vector<int> shape() const { return shpe; }
  unsigned long long size() const { return sze; }
  std::string data_type() const { return typeid(dtype).name(); }
  config::Config tensor_config() { return tensor_configuration; }

  // helper methods
  virtual bool freeze() final;  // should not be overriden
  virtual bool unfreeze() final;
  virtual tensor slice(std::initializer_list<int>);
  virtual bool reshape(std::initializer_list<int>);
  virtual bool apply_lambda(
      std::function<void(dtype)>);  // lambda or function that takes dtype and
                                    // returns nothing

  // casting elements to other dtypes
   

  //() is slicing operator
  // all operations are element-wise and final
  virtual tensor &operator()(std::initializer_list<int>);

  virtual tensor &operator+(tensor &that) final;
  virtual tensor &operator++() final;  
  virtual tensor &operator-(const tensor &that) final;
  virtual tensor &operator*(const tensor &that)final;
  virtual tensor &operator--() final; 
  virtual bool operator==(const tensor &that) final;
  virtual tensor &operator+=(const tensor &that) final;
  virtual tensor &operator-=(const tensor &that) final;
  virtual tensor &operator*=(const tensor &that) final;
};
}  // namespace tensors

#endif