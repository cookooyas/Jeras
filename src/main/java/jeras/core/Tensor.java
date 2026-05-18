package jeras.core;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import java.util.Arrays;

/**
 * 📦 [Jeras Core] 다차원 데이터 구조 및 고속 하드웨어 가속(SIMD) 연산을 담당하는 핵심 텐서 클래스
 */
public class Tensor {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public final float[] data;
    public final int[] shape; // 🌟 Keras 스타일의 다차원 형상 표현 (예: [rows, cols] 또는 [height, width, channels])
    public final int rows;
    public final int cols;

    /**
     * 표준 2차원 행렬 텐서 생성자
     */
    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.shape = new int[]{rows, cols};
        this.data = new float[rows * cols];
    }

    /**
     * 가변 차원 대응을 위한 범용 생성자 (향후 CNN 및 고급 레이어 확장 대비)
     */
    public Tensor(int... shape) {
        this.shape = shape;
        this.rows = shape.length > 0 ? shape[0] : 1;
        this.cols = shape.length > 1 ? shape[1] : 1;

        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        this.data = new float[size];
    }

    /**
     * 🌟 Keras 스타일의 텐서 형상 문자열 표현 (.shape)
     */
    public String shapeToString() {
        return Arrays.toString(shape);
    }

    /**
     * 데이터 전체를 특정 값으로 채우는 헬퍼 메서드 (예: Bias를 0.0으로 리셋할 때 사용)
     */
    public void fill(float value) {
        Arrays.fill(this.data, value);
    }

    /**
     * ⚡ [Jeras Core 가속 엔진] Vector API 기반의 초고속 행렬 곱 및 바이어스 합산 연산
     * 순전파(Forward) 시 레이어의 선형 결합(Y = XW + B)을 하드웨어 레벨에서 병렬 처리합니다.
     */
    public Tensor matMulAndAddBias(Tensor weights, Tensor bias) {
        if (this.cols != weights.rows) {
            throw new IllegalArgumentException(
                    String.format("행렬 곱 수행 불가 차원 불일치: %s x %s", this.shapeToString(), weights.shapeToString())
            );
        }

        Tensor result = new Tensor(this.rows, weights.cols);
        int upperBound = SPECIES.loopBound(weights.rows);

        for (int r = 0; r < this.rows; r++) {
            for (int c = 0; c < weights.cols; c++) {
                float sum = 0.0f;
                int i = 0;

                // 1. Vector API를 활용한 SIMD 병렬 묶음 연산 구간
                for (; i < upperBound; i += SPECIES.length()) {
                    FloatVector va = FloatVector.fromArray(SPECIES, this.data, r * this.cols + i);

                    float[] weightColumnChunk = new float[SPECIES.length()];
                    for (int vIdx = 0; vIdx < SPECIES.length(); vIdx++) {
                        weightColumnChunk[vIdx] = weights.data[(i + vIdx) * weights.cols + c];
                    }
                    FloatVector vb = FloatVector.fromArray(SPECIES, weightColumnChunk, 0);

                    sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
                }

                // 2. 남은 자투리 픽셀/가중치들의 순차 처리 구간
                for (; i < weights.rows; i++) {
                    sum += this.data[r * this.cols + i] * weights.data[i * weights.cols + c];
                }

                // 3. 결과 텐서에 최종 합산 및 바이어스 편향치 주입
                result.data[r * weights.cols + c] = sum + bias.data[c];
            }
        }
        return result;
    }
}