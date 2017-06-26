#include "layers.h"


void one_layer_test(){

    Buffer<uint8_t> data (224, 224, 3); 

    auto dataLayer = std::make_shared<DataLayerUint8>(224, 224, 3, data, "dataLayer");

    auto convLayer = new ConvUint8(4, 3, 3, 0, 1, dataLayer, "convLayer", Schedule::NPU);

    
    convLayer->getForwardFunc().print_loop_nest();
    //convLayer->getForwardFunc().trace_stores();
    
    Halide::Target target = Halide::get_host_target();
    convLayer->getForwardFunc().compile_to_lowered_stmt("conv.txt", {}, Halide::StmtOutputFormat::Text, target);

    Halide::Buffer<uint8_t> outBuff = convLayer->getForwardFunc().realize(200,200,3);
}

int main(){

    one_layer_test();

}

