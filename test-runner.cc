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

#include <stdlib.h>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "One argument of file name to compile is required.";
    return -1;
  } else {
    std::string s = "g++ -pthread ./tensors++/tests/tensor-test/";
    std::string fname(argv[1]);
    std::string output = " -o ./tensors++/tests/tensor-test/results-bin/";
    output += fname;
    output += ".out";
    std::string libdeps = " -lgtest -lgtest_main -I.";
    std::string res = (s + fname + output + libdeps);
    return std::system(res.c_str());
  }
}