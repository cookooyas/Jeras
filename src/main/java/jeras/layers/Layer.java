package jeras.layers;

import jeras.core.Tensor;

public interface Layer {
    Tensor forward(Tensor input);

    void initialize(int inputDim);

    int getOutputDim();
}