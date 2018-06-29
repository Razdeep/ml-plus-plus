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
#include <exception>
#include <string>

namespace tensors {
namespace exceptions {
class bad_cast : public std::exception {
  const char *message;
  std::string request, current;

 public:
  bad_cast(std::string s, std::string req, std::string curr)
      : message(s.c_str()), request(req), current(curr){};
  virtual const char *what() const noexcept final override {
    std::string finalized_message = std::string(message) +
                                    ". Requested to cast " + current + " to " +
                                    request + "This cast cannot be completed.";
    return finalized_message.c_str();
  };
};

class bad_reshape : public std::exception {
  const char *message;
  unsigned long long original, new_size;

 public:
  bad_reshape(std::string s, unsigned long long new_s,
              unsigned long long original_s)
      : message(s.c_str()), new_size(new_s), original(original_s){};
  virtual const char *what() const noexcept final override {
    std::string finalized_message =
        std::string(message) + ". Requested to reshape " +
        std::to_string(original) + "elements to " + std::to_string(new_size) +
        " elements. This reshape cannot be completed.";
    return finalized_message.c_str();
  };
};

class bad_slice : public std::exception {
  std::string message;

 public:
  bad_slice(std::string s) : message(s.c_str()){};
  virtual const char *what() const noexcept final override {
    std::string finalized_message =
        "Unable to slice. Invalid slicer was provided" + message;
    return finalized_message.c_str();
  };
};

class bad_indexer : public std::exception {
  std::string message;

 public:
  bad_indexer(std::string s) : message(s.c_str()){};
  virtual const char *what() const noexcept final override {
    std::string finalized_message =
        "Unable to locate. Invalid Invalid Indexer was provided" + message;
    return finalized_message.c_str();
  };
};

class broadcast_error : public std::exception {
  std::string message;

 public:
  broadcast_error(std::string s) : message(s.c_str()){};
  virtual const char *what() const noexcept final override {
    std::string finalized_message =
        "Cannot broadcast the tensor. Dimensions mismatch : " + message;
    return finalized_message.c_str();
  };
};

class operation_undefined : public std::exception {
  std::string message;

 public:
  operation_undefined(std::string s) : message(s.c_str()){};
  virtual const char *what() const noexcept final override {
    std::string finalized_message =
        "The Operation is not defined : " + message;
    return finalized_message.c_str();
  };
};

}  // namespace exceptions
}  // namespace tensors