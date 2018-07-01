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

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "tensors++/core/shape.hpp"

using namespace tensors::shape;

TEST(Dimension, SHAPE_TEST) {
  Shape s({3, 2, 4, 5});
  Shape s2({3, 2, 4});
  Shape s3({});
  EXPECT_EQ(4, s.dimension());
  EXPECT_EQ(3, s2.dimension());
  EXPECT_EQ(0, s3.dimension());
}

TEST(Element, SHAPE_TEST) {
  Shape s({3, 2, 4, 6});
  Shape s2({4, 6, 4, 46, 8, 3});
  EXPECT_EQ(3, s[0]);
  EXPECT_EQ(4, s[2]);
  EXPECT_EQ(6, s2[1]);
  EXPECT_EQ(3, s2[5]);
  EXPECT_EQ(46, s2[3]);
}

TEST(Cumulative, SHAPE_TEST) {
  Shape s({4, 1, 7, 1});
  EXPECT_EQ(4, s.cumulative_shape()[0]);
  EXPECT_EQ(4 * 1, s.cumulative_shape()[1]);
  EXPECT_EQ(4 * 1 * 7, s.cumulative_shape()[2]);
  EXPECT_EQ(4 * 1 * 7 * 1, s.cumulative_shape()[3]);
}

TEST(Equality, SHAPE_TEST) {
  Shape s({5, 6, 4});
  Shape s2({5, 6, 4});
  Shape s3({4, 5, 6});
  EXPECT_TRUE(s == s2);
  EXPECT_FALSE(s == s3);
}

TEST(SizE, SHAPE_TEST) {
  Shape s({5, 3, 6});
  EXPECT_EQ(90, s.element_size());
}

TEST(Initial_Value_Test, SHAPE_TEST) {
  std::vector<int> v = {4, -1, 9, -2};
  Shape s(v);
  EXPECT_FALSE(Shape::is_initial_valid_shape(s));
}

TEST(Stringify, SHAPE_TEST) {
  Shape s({4, 5, 3});
  EXPECT_EQ("(4, 5, 3)", static_cast<std::string>(s));
  EXPECT_EQ("(9, 5, 6, 7, 6)",
            static_cast<std::string>(Shape({9, 5, 6, 7, 6})));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
