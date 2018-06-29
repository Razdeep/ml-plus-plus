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

#ifndef TENSOR_CONFIG_HPP
#define TENSOR_CONFIG_HPP

namespace tensors {
namespace config {
struct Config {
    bool is_broad_castable; //can the tensor be broadcasted
    bool is_freezeable; //can we freeze the tensor

    static const Config default_config_instance(){
        Config cnf;
        cnf.is_freezeable = true;
        cnf.is_broad_castable = true;
        return cnf;
    }
};
}  // namespace config
}  // namespace tensors
#endif