// Halide tutorial lesson 7: Multi-stage pipelines

// On linux, you can compile and run it like so:
// g++ cnn.cpp -g -std=c++11 -I ../include -I ../tools -L ../bin -lHalide `libpng-config --cflags --ldflags` -lpthread -ldl -o cnn

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// Support code for loading pngs.
#include "halide_image_io.h"
using namespace Halide::Tools;

int main(int argc, char **argv) {
    // First we'll declare some Vars to use below.
    Var x("x"), y("y"), c("c");

    Buffer <uint8_t> img = load_image("images/rgb.png"); 

    Func input = BoundaryConditions::repeat_edge(img);
    Func input16("input_16");
    
    input16(x,y,c) = cast<uint16_t>(input(x,y,c));

    // Blur it horizontally:
    Func blur_x("blur_x");
    blur_x(x, y, c) = (input16(x-1, y, c) +
                       2 * input16(x, y, c) +
                       input16(x+1, y, c)) / 4;

    // Blur it vertically:
    Func blur_y("blur_y");
    blur_y(x, y, c) = (blur_x(x, y-1, c) +
                       2 * blur_x(x, y, c) +
                       blur_x(x, y+1, c)) / 4;

    // Convert back to 8-bit.
    Func output("output");
    output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));

    // This time it's safe to evaluate the output over the some
    // domain as the input, because we have a boundary condition.
    Buffer<uint8_t> result = output.realize(img.width(), img.height(), 3);

    // Save the result. It should look like a slightly blurry
    // parrot, but this time it will be the same size as the
    // input.
    result(x,y,c) = cast<uint8_t>(result(x,y,c));
    save_image(result, "blurry_parrot_2.png");

    printf("Success!\n");
    return 0;
}
