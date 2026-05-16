package jeras.core;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import java.util.Arrays;

public class Tensor {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public final float[] data;
    public final int rows;
    public final int cols;

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    public void initRandom() {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.random() * 0.1 - 0.05);
        }
    }

    public Tensor matMulAndAddBias(Tensor weights, Tensor bias) {
        if (this.cols != weights.rows) {
            throw new IllegalArgumentException("행렬 곱을 수행할 수 없는 차원입니다: " + this.cols + " != " + weights.rows);
        }

        Tensor result = new Tensor(this.rows, weights.cols);
        int upperBound = SPECIES.loopBound(weights.rows);

        for (int r = 0; r < this.rows; r++) {
            for (int c = 0; c < weights.cols; c++) {
                float sum = 0.0f;
                int i = 0;

                for (; i < upperBound; i += SPECIES.length()) {
                    FloatVector va = FloatVector.fromArray(SPECIES, this.data, r * this.cols + i);

                    float[] weightColumnChunk = new float[SPECIES.length()];
                    for (int vIdx = 0; vIdx < SPECIES.length(); vIdx++) {
                        weightColumnChunk[vIdx] = weights.data[(i + vIdx) * weights.cols + c];
                    }
                    FloatVector vb = FloatVector.fromArray(SPECIES, weightColumnChunk, 0);

                    sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
                }

                for (; i < weights.rows; i++) {
                    sum += this.data[r * this.cols + i] * weights.data[i * weights.cols + c];
                }

                result.data[r * weights.cols + c] = sum + bias.data[c];
            }
        }
        return result;
    }
}