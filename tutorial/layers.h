#include "Halide.h"
using namespace Halide;
Var par("par");
Var in_dim("in_dim"), n("n"), unit_dim("unit_dim");
Var x("x"), y("y"), z("z"), w("w");
Var y_t("y_t"), z_t("z_t");

class Layer {
    public:
        Layer(Layer* in) {
            // The first layer in the pipeline does not have an input layer
            if (in) {
                // Get the halide function that computes values
                // of the input layer
                assert(in->forward.defined());

                // Record the input layer
                in_layer = in;
            }
        }
        // Layer that serves as an input to the current layer
        Layer* in_layer;
        // Number of output dimensions
        virtual  int out_dims() = 0;
        // Size of output dimension i, 0 <= i < out_dims()
        virtual  int out_dim_size( int i) = 0;

        // Storage for layer parameters
        std::vector<Buffer<float> > params;
        std::vector<Buffer<float> > param_grads;
        std::vector<Buffer<float> > params_cache;
        // Halide function that computes the output of the layer
        Func forward;
        // Vector of halide functions which compute the gradients
        // with respect to layer parameters
        std::vector<Func> f_param_grads;
        // Halide function which computes gradient with respect
        // to layer input
        Func f_in_grad;
        virtual ~Layer() {};
};

class SoftMax: public Layer {
    public:
        int num_classes, num_samples;
        // Expects 2-dimensional input layer (num_classes x num_samples)
        SoftMax(Layer* in, int schedule = 1) : Layer(in) {
            assert(in->out_dims() == 2);

            forward = Func("softmax");
            Func in_f = in_layer->forward;

            num_classes = in->out_dim_size(0);
            num_samples = in->out_dim_size(1);

            // Define forward
            Func exp_max("exp_max"), expo("expo"), normalizer("normalizer");
            RDom r(0, num_classes);
            exp_max(n) = maximum(in_f(r.x, n));
            expo(in_dim, n) = exp(in_f(in_dim, n) - exp_max(n));
            normalizer(n) = cast(in_f.output_types()[0], 0);
            normalizer(n) += expo(r.x, n);
            forward(in_dim, n) = expo(in_dim, n)/normalizer(n);

            if (schedule == 1) {
                // Local schedule
                exp_max.compute_at(forward, n);
                expo.compute_at(forward, n);
                normalizer.compute_at(forward, n);
                forward.compute_root().parallel(n);
            }
        }

        // Returns a halide function that computes softmax loss given
        // the correct labels for each sample
        Func loss(Func labels) {
            // Should loss be a layer?

            // Check if labels is defined
            assert(labels.defined());
            // Check if the dimensions make sense
            assert(labels.dimensions() == 1);
            // TODO Figure out if there is a scalar type
            Func loss_p("loss_p");
            RDom r(0, num_samples);
            loss_p(x) = cast(forward.output_types()[0], 0);
            // The clamp is necessary. Otherwise, halide will assume that the
            // label can be anything during bounds inference.
            loss_p(0) += -log(forward(clamp(labels(r.x), 0, num_classes - 1),
                        r.x))/num_samples;
            return loss_p;
        }

        int out_dims() { return 2;}

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0)
                size = num_classes;
            else if (i == 1)
                size = num_samples;
            return size;
        }
};

class Affine: public Layer {
    public:
        // num_units is the number of units in the layer
        // num_inputs is the size of each input sample
        int num_units, num_samples, num_inputs;
        float reg;
        // parameters for scheduling
        Affine(int _num_units, float _reg, Layer* in,
               int schedule = 1) : Layer(in) {
            forward = Func("Affine");
            Func in_f = in_layer->forward;
            num_units = _num_units;
            reg = _reg;

            // Create parameters
            num_inputs = in->out_dim_size(0);
            num_samples = in->out_dim_size(1);

            Buffer<float> W(num_inputs, num_units), b(num_units);
            params.push_back(W); params.push_back(b);

            // Define forward
            RDom r(0, num_inputs);
            // Initialize reduction to baises
            forward(unit_dim, n) = b(unit_dim);
            // Dot product
            forward(unit_dim, n) +=
                in_f(r.x, n) * W(r.x, clamp(unit_dim, 0, num_units - 1));

            if (schedule == 1) {
                forward.compute_root().fuse(unit_dim, n, par).parallel(par);
                forward.update().fuse(unit_dim, n, par).parallel(par);
            }

        }

        int out_dims() { return 2;}

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if(i==0)
                size = num_units;
            else if(i==1)
                size = num_samples;
            return size;
        }
};

class DropOut: public Layer {
    public:
        // Threshold value between 0-1 representing the probability
        // with which a unit's output will be dropped
        float thresh;
        // Mask containing the drop out coefficients in the forward pass
        Func mask;
        DropOut(float _thresh, Layer* in) : Layer(in) {

            thresh = _thresh;

            Func in_f = in_layer->forward;

            // Define forward
            // See if there is a better way to do this
            Expr scale = 1.0f/(1.0f - thresh);
            switch(in_layer->out_dims()) {
                case 1:
                    mask(x) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x) = mask(x) * in_f(x);
                    break;
                case 2:
                    mask(x, y) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x, y) = mask(x, y) * in_f(x, y);
                    break;
                case 3:
                    mask(x, y, z) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x, y, z) = mask(x, y, z) * in_f(x, y, z);
                    break;
                case 4:
                    mask(x, y, z, w) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x, y, z, w) = mask(x, y, z, w) * in_f(x, y, z, w);
                    break;
                default:
                    assert(0);
            }
            // The mask has to be stored at root. It will be incorrect to
            // recompute the mask since the random number generator will
            // generate different values.
            mask.compute_root();
        }

        int out_dims() { return in_layer->out_dims();}

        int out_dim_size( int i) {
            return in_layer->out_dim_size(i);
        }
};

class ReLU: public Layer {
    public:
        int vec_len = 8;
        ReLU(Layer* in, int schedule = 0) : Layer(in) {
            Func in_f = in_layer->forward;
            forward = Func("ReLU");
            // Define forward
            switch(in_layer->out_dims()) {
                case 1:
                    forward(x) = max(0, in_f(x));
                    // schedule
                    if (schedule == 1) {
                        //forward.compute_root();
                        //forward.vectorize(x, vec_len);
                    }
                    break;
                case 2:
                    forward(x, y) = max(0, in_f(x, y));
                    // schedule
                    if (schedule == 1) {
                        //forward.compute_root();
                        //forward.vectorize(x, vec_len);
                        //forward.parallel(y);
                    }
                    break;
                case 3:
                    forward(x, y, z) = max(0, in_f(x, y, z));
                    // schedule
                    if (schedule == 1) {
                        //forward.compute_root();
                        //forward.vectorize(x, vec_len);
                        //forward.parallel(z);
                    }
                    break;
                case 4:
                    forward(x, y, z, w) = max(0, in_f(x, y, z, w));
                    // schedule
                    if (schedule == 1) {
                        //forward.compute_root();
                        //forward.vectorize(x, vec_len);
                        //forward.parallel(w);
                    }
                    break;
                default:
                    assert(0);
            }

        }

        int out_dims() { return in_layer->out_dims();}

        int out_dim_size( int i) {
            return in_layer->out_dim_size(i);
        }
};

class Convolutional: public Layer {
    public:
        // number of channels, height and width of the input to the layer
        int num_samples, in_ch, in_h, in_w;
        // number of filters, filter height, filter width, padding and stride
        int num_f, f_h, f_w, pad, stride;
        float reg;
        Func f_in_bound;
        // parameters for scheduling
        int o_block_size = 16;
        int y_block_size = 32;
        int vec_len = 8;
        Convolutional(int _num_f, int _f_w, int _f_h, int _pad, int _stride,
                      float _reg, Layer* in, int schedule=1) : Layer(in) {

            assert(in_layer->out_dims() == 4);
            forward = Func("conv");
            num_samples = in_layer->out_dim_size(3);
            in_ch = in_layer->out_dim_size(2);
            in_h = in_layer->out_dim_size(1);
            in_w = in_layer->out_dim_size(0);
            reg = _reg;

            assert( (in_h + 2 * _pad - _f_h) % _stride == 0);
            assert( (in_w + 2 * _pad - _f_w) % _stride == 0);

            num_f = _num_f; f_h = _f_h; f_w = _f_w;
            pad = _pad; stride = _stride;

            // Boundary condition
            // This creates a padded input and avoids checking boundary
            // conditions while computing the actual convolution
            f_in_bound = BoundaryConditions::repeat_edge(
                                    in_layer->forward,
                                    0, in_w,
                                    0, in_h);

            // Create parameters
            Buffer<float> W(f_w, f_h, in_ch, num_f), b(num_f);
            params.push_back(W); params.push_back(b);

            // Define forward
            RDom r(0, f_w, 0, f_h, 0, in_ch);
            // Initialize to bias
            forward(x, y, z, n) = b(z);
            forward(x, y, z, n) += W(r.x, r.y, r.z, z) *
                                   f_in_bound(x*stride + r.x - pad,
                                              y*stride + r.y - pad,
                                              r.z, n);

            if (schedule == 1) {
                forward.update().reorder(y, x, r.z);
                // blocking spatially with vectorization
                //f_in_bound.compute_at(f_simple, n);
                forward.compute_root();
                forward.fuse(z, n, par).parallel(par);
                forward.update().reorder(x, y, r.z);
                forward.update().split(y, y, y_t, y_block_size);
                forward.update().split(z, z, z_t, o_block_size);
                forward.update().reorder(y_t, z_t, y, r.z, z);
                forward.update().vectorize(x, vec_len);
                forward.update().fuse(z, n, par).parallel(par);
                //forward.update().fuse(y, par, par).parallel(par);
                forward.update().unroll(r.x);
                forward.update().unroll(r.y);
                // There are performance implications to this and seems to
                // be incompatible with some schedules. Have to investigate
                // this more closely.
                //f_in_bound.compute_at(forward, n);
                f_in_bound.compute_at(forward, par);
                //f_in_bound.compute_root().parallel(f_in_bound.args()[0]);
                //f_in_bound.compute_root().parallel(f_in_bound.args()[1]);
            }

        }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = (1 + (in_w + 2 * pad - f_w)/stride);
            else if (i == 1)
                size = (1 + (in_h + 2 * pad - f_h)/stride);
            else if (i == 2)
                size = num_f;
            else if (i == 3)
                size = num_samples;
            return size;
        }
};

class MaxPooling: public Layer {
    public:
        // number of color channels in input in_c
        // height and width of the input in_h, in_w
        int num_samples, in_ch, in_h, in_w;
        // height and width of the pool
        // stride at which the pooling is applied
        int p_h, p_w, stride;
        // parameters for scheduling
        int vec_len = 8;
        MaxPooling(int _p_w, int _p_h, int _stride, Layer* in,
                   int schedule = 1) : Layer(in) {
            assert(in_layer->out_dims() == 4);
            forward = Func("MaxPool");
            num_samples = in_layer->out_dim_size(3);
            in_ch = in_layer->out_dim_size(2);
            in_h = in_layer->out_dim_size(1);
            in_w = in_layer->out_dim_size(0);

            assert((in_h - _p_h) % _stride == 0);
            assert((in_w - _p_w) % _stride == 0);

            p_w = _p_w; p_h = _p_h; stride = _stride;

            // Define forward

            Func in_f = in_layer->forward;
            RDom r(0, p_w, 0, p_h);
            forward(x, y, z, n) = maximum(in_f(x * stride + r.x,
                                               y * stride + r.y,
                                               z, n));

            if (schedule == 1) {
                forward.vectorize(x, vec_len);
                forward.compute_root().fuse(z, n, par).parallel(par);
            }

        }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = 1 + ((in_w - p_w)/stride);
            else if (i == 1)
                size = 1 + ((in_h - p_h)/stride);
            else if (i == 2)
                size = in_layer->out_dim_size(2);
            else if (i == 3)
                size = num_samples;
            return size;
        }
};

class DataLayer: public Layer {
    public:
        int in_w, in_h, in_ch, num_samples;
        DataLayer(int _in_w, int _in_h, int _in_ch, int _num_samples,
                  Buffer<float> &data) : Layer(0) {
                in_w = _in_w; in_h = _in_h; in_ch = _in_ch;
                num_samples = _num_samples;
                // Define forward
                forward =
                    BoundaryConditions::repeat_edge(data, 0, in_w, 0, in_h);
        }
        // Nothing to propagate
        void back_propagate(Func dout) { assert(dout.defined()); return; }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = in_w;
            else if (i == 1)
                size = in_h;
            else if (i == 2)
                size = in_ch;
            else if (i == 3)
                size = num_samples;
            return size;
        }

};

class Flatten: public Layer {
    public:
        int out_width;
        int num_samples;
        Flatten(Layer *in, int schedule = 1) : Layer(in) {
            assert(in->out_dims() >= 2 && in->out_dims() <= 4);
            forward = Func("Flatten");
            num_samples = in_layer->out_dim_size(in_layer->out_dims() - 1);
            // Define forward
            if (in_layer->out_dims() == 2) {
                out_width = in_layer->out_dim_size(0);
                forward(x, n) = in_layer->forward(x, n);
            } else if (in_layer->out_dims() == 3) {
                int w = in_layer->out_dim_size(0);
                int h = in_layer->out_dim_size(1);
                out_width = w * h;
                forward(x, n) = in_layer->forward(x%w, (x/w), n);
            } else if (in_layer->out_dims() == 4) {
                int w = in_layer->out_dim_size(0);
                int h = in_layer->out_dim_size(1);
                int c = in_layer->out_dim_size(2);
                out_width = w * h * c;
                forward(x, n) = in_layer->forward(x%w, (x/w)%h, x/(w*h), n);
            }
            // schedule
            if (schedule == 1) {
                forward.compute_root().parallel(n);
            }

        }

        int out_dims() { return 2; }

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0)
                size = out_width;
            else if (i == 1)
                size = num_samples;
            return size;
        }
};
