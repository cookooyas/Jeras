package jeras.models;

import jeras.core.Tensor;
import jeras.layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class Sequential {
    private final List<Layer> layers = new ArrayList<>();
    private int nextInputDim;

    public Sequential(int inputShape) {
        this.nextInputDim = inputShape;
    }

    public void add(Layer layer) {
        layer.initialize(this.nextInputDim);
        layers.add(layer);
        this.nextInputDim = layer.getOutputDim();
    }

    // 순전파
    public Tensor predict(Tensor input) {
        Tensor currentOutput = input;

        for (Layer layer : layers) {
            currentOutput = layer.forward(currentOutput);
        }

        return currentOutput;
    }
}