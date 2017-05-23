#include <Halide.h>
#include <stdio.h>

int main(int argc, char **argv) {
    // Define a gradient function.
    Halide::Func f;
    Halide::Var x, y;
    f(x, y) = x + y;

    // Schedule f on the GPU in 16x16 tiles.
    f.gpu_tile(x, y, 16, 16);

    // Construct a target that uses the GPU.
    Halide::Target target = Halide::get_host_target();

    // Enable OpenCL as the GPU backend.
    //target.set_feature(Halide::Target::OpenCL);

    // Enable debugging so that you can see what OpenCL API calls we do.
    target.set_feature(Halide::Target::Debug);

    // JIT-compile the pipeline.
    //f.compile_jit(target);

    // Run it.
    //Halide::Buffer<int> result = f.realize(32, 32);
    f.compile_to_c("simple_gpu.c", {});
    f.print_loop_nest();
    
	
    // Print the result.
    //for (int y = 0; y < result.height(); y++) {
    //    for (int x = 0; x < result.width(); x++) {
    //        printf("%3d ", result(x, y));
    //    }
    //    printf("\n");
    //}

    return 0;
}
