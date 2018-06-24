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

#ifndef TENSOR_FORMATION_HPP
#define TENSOR_FORMATION_HPP

#include <exception>
#include <string>

namespace tensors {
namespace exceptions {
class freeze_exception : public std::exception {
  std::string messages;

 public:
  freeze_exception(std::string s) { messages = std::move(s); };
  virtual const char *what() const noexcept final override {
    return std::string(
               "Trying to freeze a tensor that is marked non freezeable." +
               messages)
        .c_str();
  };
};

class tensor_index_exception : public std::exception {
  const char *message;

 public:
  tensor_index_exception(std::string z) : message(z.c_str()){};
  virtual const char *what() const noexcept final override {
    return ("Index for the tensor is invalid :" + std::string(message)).c_str();
  }
};

class initializer_exception : public std::exception {
  const char *message;

 public:
  initializer_exception(std::string z) : message(z.c_str()){};
  virtual const char *what() const noexcept final override {
    return ("Unable to initialize, check that you have a valid constructor in "
            "the template type : " +
            std::string(message))
        .c_str();
  }
};

class bad_init_shape : public std::exception {
  const char *message;

 public:
  bad_init_shape(std::string z) : message(z.c_str()){};
  virtual const char *what() const noexcept final override {
    return ("Shape for Construction of Tensor is invalid. " +
            std::string(message))
        .c_str();
  }
};

}  // namespace exceptions
}  // namespace tensors

#endif
