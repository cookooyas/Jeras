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

    private Tensor weightsTensor;
    private Tensor biasTensor;

    private Tensor lastInput;
    private Tensor lastOutput; // 🌟 Leaky ReLU 역전파 정밀 미분을 위해 순전파 출력 저장용 추가
    private boolean isInitialized = false;

    /**
     * [Keras 스타일 생성자] model.add(new Dense(512, "relu")) 형태로 사용
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
        this.weightsTensor = new Tensor(inputDim, outputDim);
        this.biasTensor = new Tensor(1, outputDim);

        Random rand = new Random();
        float stdDev = (float) Math.sqrt(2.0 / inputDim);

        // He 가우시안 초기화 정밀 조율
        for (int i = 0; i < weightsTensor.data.length; i++) {
            weightsTensor.data[i] = (float) (rand.nextGaussian() * stdDev);
        }

        // Bias는 0으로 초기화
        biasTensor.fill(0.0f);

        this.isInitialized = true;
    }

    @Override
    public int getOutputDim() {
        return this.outputDim;
    }

    public Tensor getWeights() { return this.weightsTensor; }
    public Tensor getBias() { return this.biasTensor; }

    /**
     * ⚡ [Forward] 순전파 파이프라인
     */
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;

        // XW + B 선형 결합 가속 엔진
        Tensor output = input.matMulAndAddBias(this.weightsTensor, this.biasTensor);

        int batchSize = output.rows;
        if (activation.equalsIgnoreCase("relu")) {
            for (int i = 0; i < output.data.length; i++) {
                if (output.data[i] < 0) output.data[i] *= 0.01f; // Leaky ReLU 결합 유지
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

        // 🌟 역전파 활성화 미분 시 변형 전 원본 출력을 대조하기 위해 깊은 복사 보관
        this.lastOutput = new Tensor(output.rows, output.cols);
        System.arraycopy(output.data, 0, this.lastOutput.data, 0, output.data.length);

        return output;
    }

    /**
     * ⚡ [Backward] 역전파 및 가중치 조율 (오염 없는 정밀 체인 버전)
     */
    @Override
    public Tensor backward(Tensor lossGrad, float lr) {
        if (this.lastInput == null || this.lastOutput == null) {
            throw new IllegalStateException("순전파(forward)가 실행되지 않아 역전파 가중치를 계산할 수 없습니다.");
        }

        int batchSize = this.lastInput.rows;
        Tensor nextGrad = new Tensor(batchSize, inputDim);

        float[] wData = this.weightsTensor.data;
        float[] bData = this.biasTensor.data;
        float[] iData = this.lastInput.data;
        float[] oData = this.lastOutput.data;

        for (int b = 0; b < batchSize; b++) {
            int inputRowOffset = b * inputDim;
            int lossRowOffset = b * outputDim;

            for (int o = 0; o < outputDim; o++) {
                int lossIdx = lossRowOffset + o;
                float grad = lossGrad.data[lossIdx];

                // 🌟 [안정화 추가] 1. Activation (Leaky ReLU) 백프로퍼게이션 미분 필터 적용
                if (activation.equalsIgnoreCase("relu")) {
                    if (oData[lossIdx] <= 0f) {
                        grad *= 0.01f; // 순전파 때 꺾인 지점은 에러도 0.01배만 통과
                    }
                }

                // Gradient Clipping 가드
                grad = Math.max(-15.0f, Math.min(15.0f, grad));
                int weightOffset = o * inputDim;

                // 🌟 [순서 대수술] 2. 오염되지 않은 가중치 상태일 때 상위 레이어 전달용 그라디언트(nextGrad)를 먼저 연산!
                FloatVector vGrad = FloatVector.broadcast(SPECIES, grad);
                int upperBound = SPECIES.loopBound(inputDim);
                int i = 0;

                for (; i < upperBound; i += SPECIES.length()) {
                    FloatVector vw = FloatVector.fromArray(SPECIES, wData, weightOffset + i);
                    FloatVector vn = FloatVector.fromArray(SPECIES, nextGrad.data, inputRowOffset + i);

                    vn = vw.fma(vGrad, vn);
                    vn.intoArray(nextGrad.data, inputRowOffset + i);
                }

                // 나머지 테일 루프 스칼라 연산 처리
                for (; i < inputDim; i++) {
                    nextGrad.data[inputRowOffset + i] += wData[weightOffset + i] * grad;
                }

                // 🌟 3. 에러 패싱이 완벽히 끝난 후, 가중치(Weights)와 바이어스(Bias)를 안전하게 갱신
                for (int j = 0; j < inputDim; j++) {
                    wData[weightOffset + j] -= lr * grad * iData[inputRowOffset + j];
                }
                bData[o] -= lr * grad;
            }

            // 상위 레이어로 전달할 최종 결과물 배열 단위 클리핑 바인딩
            for (int i = 0; i < inputDim; i++) {
                int targetIdx = inputRowOffset + i;
                nextGrad.data[targetIdx] = Math.max(-5.0f, Math.min(5.0f, nextGrad.data[targetIdx]));
            }
        }
        return nextGrad;
    }
}