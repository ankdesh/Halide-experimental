HALIDE_SRC='/home/ankdesh/explore/Halide/Halide'
HALIDE_BUILD='/home/ankdesh/explore/Halide/build'

g++ ../lesson_03*.cpp -g -I$HALIDE_BUILD/include -L $HALIDE_BUILD/bin -lHalide -lpthread -ldl -o lesson_03 -std=c++11

# set HL_DEBUG_CODEGEN=1 or 2 to see the magic (verbose prints while codegen)
