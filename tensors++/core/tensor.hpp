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

#include <initializer_list>
#include <vector>
#include <memory>

#include "tensors++/core/tensor_config.hpp"

namespace tensors {

enum dtype { Int64, Int32, Float, Double };

class tensor {
  unsigned long long sze;
  dtype typ;
  std::vector<int> shpe;
  bool is_frozen;
  const config::Config& tensor_configuration;
  std::vector<std::vector<int>> manager;
  std::unique_ptr<void*> data;


 public:

  std::vector<int> shape() const { return shpe; }
  unsigned long long size() const { return sze; }
  dtype data_type() const { return typ; }
  config::Config tensor_config() { return tensor_configuration; }

  virtual bool freeze() final; //should not be overriden
  virtual bool unfreeze() final;
  virtual tensor slice(std::initializer_list<int>);
  virtual bool reshape(std::initializer_list<int>);


  //this is slicing operator
  //all operations are element-wise
  virtual tensor operator()(int points...); 
  virtual tensor operator+(tensor &other);
  virtual tensor operator++(int); //post
  virtual tensor operator-(tensor &other);
  virtual tensor operator*(tensor &other);
  virtual tensor operator--(int); //post


};

}  // namespace tensors

#endif