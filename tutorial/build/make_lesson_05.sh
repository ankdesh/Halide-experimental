HALIDE_SRC='/home/ankdesh/explore/Halide/Halide'
HALIDE_BUILD='/home/ankdesh/explore/Halide/build'

g++ ../lesson_05*.cpp -g -I$HALIDE_BUILD/include -L $HALIDE_BUILD/bin -lHalide -lpthread -ldl -o lesson_05 -std=c++11
