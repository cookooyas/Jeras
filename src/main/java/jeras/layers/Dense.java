package jeras.layers;

import jeras.core.Tensor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import java.util.Random;

public class Dense implements Layer {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private int inputDim;   // 🌟 Keras 스타일을 위해 final 제거 (initialize 시점에 할당 가능하도록)
    private final int outputDim;
    private final String activation;

    public float[] weights;
    public float[] bias;
    private boolean isInitialized = false; // 가중치 중복 초기화 방지 플래그

    /**
     * [생성자 1] 🌟 JerasKerasStyleTest 호환용 (인수 2개: Keras 스타일)
     * 입력 차원은 Sequential.add() 내부에서 layer.initialize()가 호출될 때 동적으로 세팅됩니다.
     */
    public Dense(int outputDim, String activation) {
        this.outputDim = outputDim;
        this.activation = activation;
    }

    /**
     * [생성자 2] MnistCompleteTrainingTest 호환용 (인수 3개: 명시적 스타일)
     */
    public Dense(int inputDim, int outputDim, String activation) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.activation = activation;
        allocateAndInitializeWeights();
    }

    /**
     * Sequential.add() 시점에 프레임워크가 앞 레이어의 차원을 주입하는 곳
     */
    @Override
    public void initialize(int inputDim) {
        if (!isInitialized) {
            this.inputDim = inputDim;
            allocateAndInitializeWeights();
        } else if (this.inputDim != inputDim) {
            System.out.println("⚠️ [Jeras 경고] 입력 차원 불일치 감지: " + this.inputDim + " vs " + inputDim);
        }
    }

    /**
     * 🌟 실제 메모리를 할당하고 He 가중치 초기화를 수행하는 핵심 메서드
     */
    private void allocateAndInitializeWeights() {
        this.weights = new float[inputDim * outputDim];
        this.bias = new float[outputDim];

        Random rand = new Random();
        float stdDev = (float) Math.sqrt(2.0 / inputDim);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (rand.nextGaussian() * stdDev);
        }
        this.isInitialized = true;
    }

    @Override
    public int getOutputDim() {
        return this.outputDim;
    }

    public float[] getWeights() { return this.weights; }
    public float[] getBias() { return this.bias; }

    /**
     * SIMD 가속 순전파 및 Safe Activation
     */
    @Override
    public Tensor forward(Tensor input) {
        int batchSize = input.rows;
        Tensor output = new Tensor(batchSize, outputDim);

        for (int b = 0; b < batchSize; b++) {
            int inputRowOffset = b * inputDim;
            int outputRowOffset = b * outputDim;

            for (int o = 0; o < outputDim; o++) {
                int weightOffset = o * inputDim;

                FloatVector acc = FloatVector.zero(SPECIES);
                int upperBound = SPECIES.loopBound(inputDim);
                int i = 0;

                for (; i < upperBound; i += SPECIES.length()) {
                    FloatVector vx = FloatVector.fromArray(SPECIES, input.data, inputRowOffset + i);
                    FloatVector vw = FloatVector.fromArray(SPECIES, weights, weightOffset + i);
                    acc = vx.fma(vw, acc);
                }

                float sum = acc.reduceLanes(VectorOperators.ADD);

                for (; i < inputDim; i++) {
                    sum += input.data[inputRowOffset + i] * weights[weightOffset + i];
                }

                output.data[outputRowOffset + o] = sum + bias[o];
            }
        }

        if (activation.equalsIgnoreCase("relu")) {
            for (int i = 0; i < output.data.length; i++) {
                output.data[i] = Math.max(0.0f, output.data[i]);
            }
        } else if (activation.equalsIgnoreCase("softmax")) {
            for (int b = 0; b < batchSize; b++) {
                int offset = b * outputDim;
                float maxVal = output.data[offset];
                for (int o = 1; o < outputDim; o++) {
                    if (output.data[offset + o] > maxVal) maxVal = output.data[offset + o];
                }

                float expSum = 0.0f;
                for (int o = 0; o < outputDim; o++) {
                    output.data[offset + o] = (float) Math.exp(output.data[offset + o] - maxVal);
                    expSum += output.data[offset + o];
                }

                for (int o = 0; o < outputDim; o++) {
                    output.data[offset + o] = output.data[offset + o] / (expSum + 1e-9f);
                    output.data[offset + o] = Math.max(1e-7f, Math.min(1.0f - 1e-7f, output.data[offset + o]));
                }
            }
        }
        return output;
    }

    /**
     * SIMD 가속 역전파
     */
    @Override
    public Tensor backward(Tensor input, Tensor lossGrad, float lr) {
        int batchSize = input.rows;
        Tensor nextGrad = new Tensor(batchSize, inputDim);

        for (int b = 0; b < batchSize; b++) {
            int inputRowOffset = b * inputDim;
            int lossRowOffset = b * outputDim;

            for (int o = 0; o < outputDim; o++) {
                float grad = lossGrad.data[lossRowOffset + o];
                int weightOffset = o * inputDim;

                for (int i = 0; i < inputDim; i++) {
                    weights[weightOffset + i] -= lr * grad * input.data[inputRowOffset + i];
                }
                bias[o] -= lr * grad;

                FloatVector vGrad = FloatVector.broadcast(SPECIES, grad);
                int upperBound = SPECIES.loopBound(inputDim);
                int i = 0;

                for (; i < upperBound; i += SPECIES.length()) {
                    FloatVector vw = FloatVector.fromArray(SPECIES, weights, weightOffset + i);
                    FloatVector vn = FloatVector.fromArray(SPECIES, nextGrad.data, inputRowOffset + i);
                    vn = vw.fma(vGrad, vn);
                    vn.intoArray(nextGrad.data, inputRowOffset + i);
                }

                for (; i < inputDim; i++) {
                    nextGrad.data[inputRowOffset + i] += weights[weightOffset + i] * grad;
                }
            }
        }
        return nextGrad;
    }
}