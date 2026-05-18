package jeras.layers;

import jeras.core.Tensor;

public interface Layer {
    void initialize(int inputDim);
    int getOutputDim();
    Tensor forward(Tensor input);
    Tensor backward(Tensor input, Tensor lossGrad, float lr);
}