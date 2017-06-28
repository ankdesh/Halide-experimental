#include "Halide.h"
using namespace Halide;
Var par("par");
Var in_dim("in_dim"), n("n"), unit_dim("unit_dim");
Var x("x"), y("y"), z("z"), w("w");
Var y_t("y_t"), z_t("z_t");

enum class Schedule{
  CPU,
  GPU,
  NPU
};

template <typename T>
class Layer {
    public:
        Layer(std::shared_ptr<Layer> in, std::string layerName) {
            // The first layer in the pipeline does not have an input layer
            if (inLayer) {
                // Get the halide function that computes values
                // of the input layer
                assert(in->getForwardFunc().defined());

                // Record the input layer
                inLayer = in;
            }
            forward = Func(layerName);
        }

        // Number of output dimensions
        virtual  int out_dims() = 0;

        // Size of output dimension i, 0 <= i < out_dims()
        virtual  int out_dim_size( int i) = 0;

        // Gets Forward Layer ref
        Func& getForwardFunc(){
            return forward;
        }

        protected:
        // Layer that serves as an input to the current layer
        std::shared_ptr<Layer> inLayer;

        // Storage for layer parameters
        std::vector<Buffer<T> > params;

        // Halide function that computes the output of the layer
        Func forward;

        virtual ~Layer() {};
};

class ConvUint8: public Layer<uint8_t> {
    public:
        ConvUint8(int numF, 
                  int fWidth, 
                  int fHeight, 
                  int pad, 
                  int stride,
                  std::shared_ptr<Layer> inLayer,
                  std::string layerName, 
                  Schedule sched = Schedule::CPU) : 
                    Layer     (inLayer, layerName),
                    ifmHeight (inLayer->out_dim_size(0)),
                    ifmWidth  (inLayer->out_dim_size(1)),
                    ifmDepth  (inLayer->out_dim_size(2)),
                    fNum      (numF),
                    fHeight   (fWidth),
                    fWidth    (fHeight),
                    pad       (pad),
                    stride    (stride),
                    sched     (sched) {

            assert(inLayer->out_dims() == 3);

            // Boundary condition
            // This creates a padded input and avoids checking boundary
            // conditions while computing the actual convolution
            //f_in_bound = BoundaryConditions::repeat_edge(
            //                        inLayer->getForwardFunc(),
            //                        0, ifmWidth,
            //                        0, ifmHeight);

            // Create parameters
            Buffer<uint8_t> W(fWidth, fHeight, ifmDepth, numF), b(numF);
            params.push_back(W); params.push_back(b);
            // TODO ankdesh Read Weights from mem

            // Define forward
            RDom r(0, fWidth, 0, fHeight, 0, ifmDepth);
           
            Func& inData = inLayer->getForwardFunc();
            forward(x, y, z) = sum(W(r.x, r.y, r.z, z) * inData(x * stride + r.x - pad, y * stride + r.y - pad , z));
 
            // Initialize to bias
            //forward(x, y, z) = b(z);
            //forward(x, y, z) += W(r.x, r.y, r.z, z) *
            //                       inLayer->getForwardFunc()(x*stride + r.x - pad,
            //                                  y*stride + r.y - pad,
            //                                  r.z);
            // Set the schedule
            setDefaultSchedule_(sched, r);
        }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = (1 + (ifmWidth + 2 * pad - fWidth)/stride);
            else if (i == 1)
                size = (1 + (ifmHeight + 2 * pad - fHeight)/stride);
            else if (i == 2)
                size = fNum;
            return size;
        }
        
        private:
        // number of channels, height and width of the input to the layer
        int ifmHeight;
        int ifmWidth;
        int ifmDepth;
        
        // number of filters, filter height, filter width, padding and stride
        int fNum;
        int fHeight;
        int fWidth;
        int pad;
        int stride;
      
        // Schdule
        Schedule sched;

        Func f_in_bound;

        void setDefaultSchedule_(Schedule sched, RDom& r){
            if (sched == Schedule::CPU) {
                // parameters for scheduling
                int o_block_size = 16;
                int y_block_size = 32;
                int vec_len = 8;
        
                forward.update().reorder(y, x, r.z);
                forward.compute_root();
                forward.update().reorder(x, y, r.z);
                forward.update().split(y, y, y_t, y_block_size);
                forward.update().split(z, z, z_t, o_block_size);
                forward.update().reorder(y_t, z_t, y, r.z, z);
                forward.update().vectorize(x, vec_len);
                forward.update().unroll(r.x);
                forward.update().unroll(r.y);
                f_in_bound.compute_at(forward, par);
            }
            else if (sched == Schedule::NPU){
                Var xi,yi,zi;
                forward.split(z,z,zi,32);
                forward.tile(x,y,xi,yi,64,64);
                forward.reorder(yi,xi,zi,z,y,x);
            }
        }

};

class DataLayerUint8: public Layer<uint8_t> {
    public:
        DataLayerUint8
            (int ifmWidth, 
            int ifmHeight, 
            int ifmDepth,
            Buffer<uint8_t>& data,
            std::string layerName) : 
                Layer(0, layerName),
                ifmWidth (ifmWidth),
                ifmHeight (ifmHeight),
                ifmDepth (ifmDepth) {
            
                // Define forward
                //forward =
                // BoundaryConditions::repeat_edge(data, 0, ifmWidth, 0, ifmHeight);
                forward (x,y,z) = data (x,y,z);
        }

        int out_dims() { return 3; }

        int out_dim_size( int i) {
            assert(i < 3);
            int size = 0;
            if (i == 0)
                size = ifmWidth;
            else if (i == 1)
                size = ifmHeight;
            else if (i == 2)
                size = ifmDepth;
            return size;
        }
    
        private:
        int ifmWidth; 
        int ifmHeight; 
        int ifmDepth; 
};

