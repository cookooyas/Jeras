package jeras.layers;

import jeras.core.Tensor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import java.util.Random;

/**
 * 🧱 [Jeras Layer] 완전 연결 레이어 (Keras의 layers.Dense와 1:1 대응)
 * 내부적으로 Vector API(SIMD)를 이용한 오차 역전파 및 가중치 업데이트가 자동 캡슐화되어 작동합니다.
 */
public class Dense implements Layer {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private int inputDim;
    private final int outputDim;
    private final String activation;

    // 🌟 가중치와 바이어스를 순수 배열에서 명품 Tensor 객체로 승격
    private Tensor weightsTensor;
    private Tensor biasTensor;

    // 🌟 역전파를 위해 순전파 시 들어온 입력을 레이어 내부에 캐싱 (Keras의 연산 그래프 철학)
    private Tensor lastInput;
    private boolean isInitialized = false;

    /**
     * [Keras 스타일 생성자] model.add(new Dense(512, "relu")) 형태로 사용
     * 입력 차원은 Sequential 파이프라인에 연결될 때 자동으로 빌드됩니다.
     */
    public Dense(int outputDim, String activation) {
        this.outputDim = outputDim;
        this.activation = activation;
    }

    /**
     * [명시적 생성자] 첫 번째 레이어에서 명시적으로 차원을 지정할 때 사용
     */
    public Dense(int inputDim, int outputDim, String activation) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.activation = activation;
        allocateAndInitializeWeights();
    }

    @Override
    public void initialize(int inputDim) {
        if (!isInitialized) {
            this.inputDim = inputDim;
            allocateAndInitializeWeights();
        }
    }

    /**
     * 🧠 He Normal (MSRA) 가중치 초기화 알고리즘 기반 텐서 메모리 할당
     */
    private void allocateAndInitializeWeights() {
        // 행렬 곱 연산 방향에 맞춰 가중치 형상을 [inputDim, outputDim]으로 세팅
        this.weightsTensor = new Tensor(inputDim, outputDim);
        this.biasTensor = new Tensor(1, outputDim); // 바이어스는 출력 차원 크기만큼 전개

        Random rand = new Random();
        float stdDev = (float) Math.sqrt(2.0 / inputDim);

        // He 가우시안 초기화 주입
        for (int i = 0; i < weightsTensor.data.length; i++) {
            weightsTensor.data[i] = (float) (rand.nextGaussian() * stdDev);
        }
        // Bias는 Keras 표준에 따라 초기값 0.0으로 맑게 밀어버립니다.
        biasTensor.fill(0.0f);

        this.isInitialized = true;
    }

    @Override
    public int getOutputDim() {
        return this.outputDim;
    }

    // 외부 노출 API도 Tensor 객체 타겟으로 래핑
    public Tensor getWeights() { return this.weightsTensor; }
    public Tensor getBias() { return this.biasTensor; }

    /**
     * ⚡ [Forward] 순전파 파이프라인
     */
    @Override
    public Tensor forward(Tensor input) {
        // 🌟 [핵심] 오차 역전파 체인을 위해 현재 들어온 입력을 내부에 캐싱합니다.
        this.lastInput = input;

        // 🌟 복잡한 하드웨어 가속 선형 결합(XW + B) 연산은 Tensor의 고유 코어 엔진에 완전히 위임!
        Tensor output = input.matMulAndAddBias(this.weightsTensor, this.biasTensor);

        // 활성화 함수 처리 블록
        int batchSize = output.rows;
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
     * ⚡ [Backward] 역전파 및 가중치 조율
     * 🌟 캐싱된 lastInput을 사용하므로, 외부 지휘관(Sequential)은 인수를 덕지덕지 넘길 필요가 없어집니다.
     */
    @Override
    public Tensor backward(Tensor lossGrad, float lr) {
        if (this.lastInput == null) {
            throw new IllegalStateException("순전파(forward)가 실행되지 않아 역전파 가중치를 계산할 수 없습니다.");
        }

        int batchSize = this.lastInput.rows;
        Tensor nextGrad = new Tensor(batchSize, inputDim);

        float[] wData = this.weightsTensor.data;
        float[] bData = this.biasTensor.data;
        float[] iData = this.lastInput.data;

        for (int b = 0; b < batchSize; b++) {
            int inputRowOffset = b * inputDim;
            int lossRowOffset = b * outputDim;

            for (int o = 0; o < outputDim; o++) {
                float grad = lossGrad.data[lossRowOffset + o];
                int weightOffset = o * inputDim;

                // 1. 가중치(Weights) 및 바이어스(Bias) 미세 조율
                for (int i = 0; i < inputDim; i++) {
                    wData[weightOffset + i] -= lr * grad * iData[inputRowOffset + i];
                }
                bData[o] -= lr * grad;

                // 2. Vector API 가속을 통한 상위 레이어 전달용 그라디언트 연산
                FloatVector vGrad = FloatVector.broadcast(SPECIES, grad);
                int upperBound = SPECIES.loopBound(inputDim);
                int i = 0;

                for (; i < upperBound; i += SPECIES.length()) {
                    FloatVector vw = FloatVector.fromArray(SPECIES, wData, weightOffset + i);
                    FloatVector vn = FloatVector.fromArray(SPECIES, nextGrad.data, inputRowOffset + i);
                    vn = vw.fma(vGrad, vn);
                    vn.intoArray(nextGrad.data, inputRowOffset + i);
                }

                for (; i < inputDim; i++) {
                    nextGrad.data[inputRowOffset + i] += wData[weightOffset + i] * grad;
                }
            }
        }
        return nextGrad;
    }
}