package jeras.layers;

import jeras.core.Tensor;

import java.util.Arrays;

public class Dense implements Layer {
    private final int units;
    private final String activation;
    private Tensor weights;
    private Tensor bias;
    private int inputDim;

    public Dense(int units, String activation) {
        this.units = units;
        this.activation = activation.toLowerCase();
    }

    @Override
    public void initialize(int inputDim) {
        this.inputDim = inputDim;
        this.weights = new Tensor(inputDim, units);
        this.weights.initRandom();

        this.bias = new Tensor(1, units);
        Arrays.fill(this.bias.data, 0.01f);
    }

    @Override
    public Tensor forward(Tensor input) {
        // Vector API 실행부
        Tensor output = input.matMulAndAddBias(this.weights, this.bias);

        // 활성화 함수 통과계층 (relu, softmax)
        if ("relu".equals(activation)) {
            for (int i = 0; i < output.data.length; i++) {
                if (output.data[i] < 0) {
                    output.data[i] = 0f;
                }
            }
        } else if ("softmax".equals(activation)) {
            for (int r = 0; r < output.rows; r++) {
                int rowOffset = r * output.cols;
                float max = output.data[rowOffset];
                for (int c = 1; c < output.cols; c++) {
                    max = Math.max(max, output.data[rowOffset + c]);
                }

                float sum = 0f;
                for (int c = 0; c < output.cols; c++) {
                    output.data[rowOffset + c] = (float) Math.exp(output.data[rowOffset + c] - max);
                    sum += output.data[rowOffset + c];
                }

                for (int c = 0; c < output.cols; c++) {
                    output.data[rowOffset + c] /= sum;
                }
            }
        }
        return output;
    }

    // 역전파
    public Tensor backward(Tensor input, Tensor gradient, float learningRate) {
        Tensor weightGrad = new Tensor(this.inputDim, this.units);

        // 경사하강법 기반 오차수정
        for (int r = 0; r < this.inputDim; r++) {
            for (int c = 0; c < this.units; c++) {
                float sumGrad = 0f;
                for (int b = 0; b < input.rows; b++) {
                    sumGrad += input.data[b * this.inputDim + r] * gradient.data[b * this.units + c];
                }
                // 가중치 업데이트
                this.weights.data[r * this.units + c] -= learningRate * sumGrad;
            }
        }

        // 편향성
        for (int c = 0; c < this.units; c++) {
            float sumGrad = 0f;
            for (int b = 0; b < input.rows; b++) {
                sumGrad += gradient.data[b * this.units + c];
            }
            this.bias.data[c] -= learningRate * sumGrad;
        }

        Tensor inputGrad = new Tensor(input.rows, this.inputDim);
        for (int b = 0; b < input.rows; b++) {
            for (int r = 0; r < this.inputDim; r++) {
                float sum = 0f;
                for (int c = 0; c < this.units; c++) {
                    sum += gradient.data[b * this.units + c] * this.weights.data[r * this.units + c];
                }
                inputGrad.data[b * this.inputDim + r] = sum;
            }
        }

        return inputGrad;
    }

    @Override
    public int getOutputDim() {
        return this.units;
    }
}