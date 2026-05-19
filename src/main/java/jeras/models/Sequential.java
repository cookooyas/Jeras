package jeras.models;

import jeras.core.Tensor;
import jeras.layers.Layer;
import jeras.utils.Augmentation;
import jeras.utils.DataLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 🎬 [Jeras 무한 유연성 사령탑 - 가중치 파괴 버그 수정본]
 */
public class Sequential {
    private final List<Layer> layers = new ArrayList<>();
    private final int inputDim;
    private float learningRate;

    private final List<float[]> weightsList = new ArrayList<>();
    private final List<float[]> biasesList = new ArrayList<>();

    private final List<Integer> inputDims = new ArrayList<>();
    private final List<Integer> outputDims = new ArrayList<>();

    public Sequential(int inputShape) {
        this.inputDim = inputShape;
    }

    public void add(Layer layer) {
        layers.add(layer);

        int currentInDim = inputDims.isEmpty() ? this.inputDim : outputDims.get(outputDims.size() - 1);
        int currentOutDim = layer.getOutputDim();

        inputDims.add(currentInDim);
        outputDims.add(currentOutDim);

        float[] w = new float[currentInDim * currentOutDim];
        float[] b = new float[currentOutDim];

        Random rand = new Random();
        float stdDev = (float) Math.sqrt(2.0 / currentInDim);
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (rand.nextGaussian() * stdDev);
        }
        // 바이어스는 안전하게 0.0f 초기화
        for (int i = 0; i < b.length; i++) b[i] = 0.0f;

        weightsList.add(w);
        biasesList.add(b);
    }

    public void compile(float learningRate, String loss, String... metrics) {
        this.learningRate = learningRate;
    }

    /**
     * 🏋️ [모드 A] 원샷 통짜 배열 방식 (Array Mode)
     */
    public void fit(float[] rawX, float[] rawY, int batchSize, int epochs) {
        int totalSamples = rawX.length / inputDim;
        int numBatches = totalSamples / batchSize;
        int finalOutDim = outputDims.get(outputDims.size() - 1);
        Augmentation codeAugmenter = new Augmentation();

        System.out.printf("Train on %d samples (Dynamic Pure Engine - Array Mode)%n", totalSamples);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            float epochLoss = 0f;
            int correctCount = 0;

            System.out.printf("Epoch %d/%d%n", epoch, epochs);

            for (int b = 0; b < numBatches; b++) {
                float[] batchX = new float[batchSize * inputDim];
                float[] batchY = new float[batchSize * finalOutDim];

                for (int s = 0; s < batchSize; s++) {
                    int sampleIdx = b * batchSize + s;
                    float[] singleImage = new float[inputDim];
                    System.arraycopy(rawX, sampleIdx * inputDim, singleImage, 0, inputDim);

                    float[] augmentedImage = codeAugmenter.augment(singleImage);
                    System.arraycopy(augmentedImage, 0, batchX, s * inputDim, inputDim);
                    System.arraycopy(rawY, sampleIdx * finalOutDim, batchY, s * finalOutDim, finalOutDim);
                }

                InnerBatchResult res = trainSingleBatch(batchX, batchY, batchSize);
                epochLoss += res.loss;
                correctCount += res.correct;

                if ((b + 1) % (numBatches / 10) == 0 || b == numBatches - 1) {
                    printProgressBar(b + 1, numBatches);
                }
            }
            printEpochSummary(epochStartTime, epochLoss, numBatches, correctCount, totalSamples);
        }
    }

    /**
     * 🏋️ [모드 B] 가상 스레드 DataLoader 전용 오버로딩 (Virtual Thread Mode)
     */
    public void fit(DataLoader dataLoader, int numBatches, int epochs) {
        System.out.printf("Train on Jeras Async Pipeline (Dynamic Pure Engine - Virtual Thread Mode)%n");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            float epochLoss = 0f;
            int correctCount = 0;
            int totalSamplesEvaluated = 0;

            System.out.printf("Epoch %d/%d%n", epoch, epochs);

            for (int b = 0; b < numBatches; b++) {
                try {
                    DataLoader.DataBatch batch = dataLoader.nextBatch();
                    Tensor tensorX = batch.x();
                    Tensor tensorY = batch.y();

                    int batchSize = tensorX.rows;
                    totalSamplesEvaluated += batchSize;

                    // 🛠️ 불필요한 복사 코드를 전면 걷어내고 큐에서 나온 데이터를 순수하게 직주입
                    InnerBatchResult res = trainSingleBatch(tensorX.data, tensorY.data, batchSize);
                    epochLoss += res.loss;
                    correctCount += res.correct;

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    System.out.println("\n⚠️ [Jeras 경고] 데이터 로더 인터럽트 감지.");
                    return;
                }

                if ((b + 1) % (numBatches / 10) == 0 || b == numBatches - 1) {
                    printProgressBar(b + 1, numBatches);
                }
            }
            printEpochSummary(epochStartTime, epochLoss, numBatches, correctCount, totalSamplesEvaluated);
        }
    }

    /**
     * 🧠 [순정 동적 연산 엔진]
     */
    private InnerBatchResult trainSingleBatch(float[] batchX, float[] batchY, int batchSize) {
        int totalLayers = weightsList.size();

        List<float[]> linearsCache = new ArrayList<>();
        List<float[]> outputsCache = new ArrayList<>();

        float[] currentInput = batchX;
        int currentInDim = this.inputDim;

        // 🚀 1. FORWARD CHAIN
        for (int l = 0; l < totalLayers; l++) {
            int outDim = outputDims.get(l);
            float[] wData = weightsList.get(l);
            float[] bData = biasesList.get(l);

            float[] linear = new float[batchSize * outDim];
            float[] output = new float[batchSize * outDim];

            for (int s = 0; s < batchSize; s++) {
                int sInOffset = s * currentInDim;
                int sOutOffset = s * outDim;

                for (int o = 0; o < outDim; o++) {
                    float sum = bData[o];
                    int wOffset = o * currentInDim;
                    for (int i = 0; i < currentInDim; i++) {
                        sum += currentInput[sInOffset + i] * wData[wOffset + i];
                    }
                    linear[sOutOffset + o] = sum;

                    if (l < totalLayers - 1) {
                        output[sOutOffset + o] = sum > 0 ? sum : sum * 0.01f; // Leaky ReLU
                    } else {
                        output[sOutOffset + o] = sum;
                    }
                }
            }
            linearsCache.add(linear);
            outputsCache.add(output);

            currentInput = output;
            currentInDim = outDim;
        }

        // 🚀 2. SOFTMAX LOSS CALCULATOR (수치 안정성 강화)
        int finalOutDim = outputDims.get(totalLayers - 1);
        float[] lastLogits = linearsCache.get(totalLayers - 1);
        float[] lossGrad = new float[batchSize * finalOutDim];
        float batchLossSum = 0f;
        int correct = 0;

        for (int s = 0; s < batchSize; s++) {
            int outOffset = s * finalOutDim;
            float maxLogit = -Float.MAX_VALUE;
            for (int o = 0; o < finalOutDim; o++) {
                if (lastLogits[outOffset + o] > maxLogit) maxLogit = lastLogits[outOffset + o];
            }

            float expSum = 0f;
            float[] probs = new float[finalOutDim];
            for (int o = 0; o < finalOutDim; o++) {
                probs[o] = (float) Math.exp(lastLogits[outOffset + o] - maxLogit);
                expSum += probs[o];
            }

            int maxPredictIdx = 0;
            int maxTargetIdx = 0;
            float maxProb = -1f;

            for (int o = 0; o < finalOutDim; o++) {
                probs[o] /= (expSum + 1e-9f);

                if (batchY[outOffset + o] == 1.0f) {
                    // 🌟 수치 안정 가드 범위 대폭 확장하여 NaN 원천 차단
                    batchLossSum += (float) -Math.log(Math.max(probs[o], 1e-15f));
                    maxTargetIdx = o;
                }

                lossGrad[outOffset + o] = probs[o] - batchY[outOffset + o];

                if (probs[o] > maxProb) {
                    maxProb = probs[o];
                    maxPredictIdx = o;
                }
            }
            if (maxPredictIdx == maxTargetIdx) correct++;
        }

        // 🚀 3. BACKPROPAGATION & WEIGHT UPDATE CHAIN
        float[] currentLossGrad = lossGrad;

        for (int l = totalLayers - 1; l >= 0; l--) {
            int inDim = inputDims.get(l);
            int outDim = outputDims.get(l);
            float[] wData = weightsList.get(l);
            float[] bData = biasesList.get(l);

            float[] lLinear = linearsCache.get(l);
            float[] lInput = (l == 0) ? batchX : outputsCache.get(l - 1);

            float[] nextLayerGrad = new float[batchSize * inDim];

            for (int s = 0; s < batchSize; s++) {
                int sInOffset = s * inDim;
                int sOutOffset = s * outDim;

                for (int o = 0; o < outDim; o++) {
                    float grad = currentLossGrad[sOutOffset + o];

                    if (l < totalLayers - 1) {
                        if (lLinear[sOutOffset + o] <= 0) grad *= 0.01f;
                    }

                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++) {
                        nextLayerGrad[sInOffset + i] += wData[wOffset + i] * grad;
                    }
                    for (int i = 0; i < inDim; i++) {
                        wData[wOffset + i] -= learningRate * grad * lInput[sInOffset + i];
                    }
                    bData[o] -= learningRate * grad;
                }
            }
            currentLossGrad = nextLayerGrad;
        }

        return new InnerBatchResult(batchLossSum / batchSize, correct);
    }

    /**
     * 🎯 [단일 이미지 동적 추론]
     */
    public float[] predict(float[] singleImage) {
        if (weightsList.isEmpty()) {
            throw new IllegalStateException("가중치 인프라가 구축되지 않아 추론을 시작할 수 없습니다.");
        }

        int totalLayers = weightsList.size();
        float[] currentInput = singleImage;

        for (int l = 0; l < totalLayers; l++) {
            int inDim = inputDims.get(l);
            int outDim = outputDims.get(l);
            float[] wData = weightsList.get(l);
            float[] bData = biasesList.get(l);

            float[] output = new float[outDim];

            for (int o = 0; o < outDim; o++) {
                float sum = bData[o];
                int wOffset = o * inDim;
                for (int i = 0; i < inDim; i++) {
                    sum += currentInput[i] * wData[wOffset + i];
                }
                if (l < totalLayers - 1) {
                    output[o] = sum > 0 ? sum : sum * 0.01f; // Leaky ReLU
                } else {
                    output[o] = sum;
                }
            }
            currentInput = output;
        }

        float[] logits = currentInput;
        float maxLogit = -Float.MAX_VALUE;
        for (float v : logits) if (v > maxLogit) maxLogit = v;

        float expSum = 0f;
        float[] probs = new float[logits.length];
        for (int o = 0; o < logits.length; o++) {
            probs[o] = (float) Math.exp(logits[o] - maxLogit);
            expSum += probs[o];
        }
        for (int o = 0; o < logits.length; o++) probs[o] /= (expSum + 1e-9f);

        return probs;
    }

    public void saveWeights(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (int l = 0; l < weightsList.size(); l++) {
                writer.write("LAYER_" + l + "_WEIGHTS\n");
                float[] w = weightsList.get(l);
                for (float v : w) writer.write(v + ",");

                writer.write("\nLAYER_" + l + "_BIAS\n");
                float[] b = biasesList.get(l);
                for (float v : b) writer.write(v + ",");
                writer.write("\n");
            }
        }
    }

    public void loadWeights(String filePath) throws IOException {
        weightsList.clear();
        biasesList.clear();
        inputDims.clear();
        outputDims.clear();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.endsWith("_WEIGHTS")) {
                    String dataLine = reader.readLine();
                    if (dataLine != null && !dataLine.trim().isEmpty()) {
                        String[] tokens = dataLine.split(",");
                        int totalElements = tokens.length;

                        int currentInDim = inputDims.isEmpty() ? this.inputDim : outputDims.get(outputDims.size() - 1);
                        int currentOutDim = totalElements / currentInDim;

                        inputDims.add(currentInDim);
                        outputDims.add(currentOutDim);

                        weightsList.add(new float[totalElements]);
                        biasesList.add(new float[currentOutDim]);
                    }
                }
            }
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int currentLayerIdx = -1;
            boolean isWeightMode = true;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                if (line.contains("_WEIGHTS")) {
                    currentLayerIdx = Integer.parseInt(line.replaceAll("[^0-9]", ""));
                    isWeightMode = true;
                    continue;
                }
                if (line.contains("_BIAS")) {
                    currentLayerIdx = Integer.parseInt(line.replaceAll("[^0-9]", ""));
                    isWeightMode = false;
                    continue;
                }

                float[] targetArr = isWeightMode ? weightsList.get(currentLayerIdx) : biasesList.get(currentLayerIdx);
                String[] tokens = line.split(",");
                int idx = 0;
                for (String token : tokens) {
                    if (!token.trim().isEmpty() && idx < targetArr.length) {
                        targetArr[idx++] = Float.parseFloat(token);
                    }
                }
            }
        }
        System.out.printf("[Jeras Engine] ⚙️ 복원 완료: 총 %d개의 가변 레이어 구조를 복원했습니다!%n", weightsList.size());
    }

    private static record InnerBatchResult(float loss, int correct) {}

    private void printProgressBar(int current, int total) {
        int barLength = 30;
        int progress = (int) ((float) current / total * barLength);
        StringBuilder bar = new StringBuilder();
        for (int i = 0; i < barLength; i++) {
            if (i < progress) bar.append("=");
            else if (i == progress) bar.append(">");
            else bar.append(".");
        }
        System.out.printf("\r%d/%d [%s]", current, total, bar.toString());
    }

    private void printEpochSummary(long startTime, float epochLoss, int numBatches, int correctCount, int totalSamples) {
        long epochTime = (System.currentTimeMillis() - startTime) / 1000;
        float finalLoss = epochLoss / numBatches;
        float finalAcc = (float) correctCount / totalSamples;
        System.out.printf(" - %ds - loss: %.4f - accuracy: %.4f%n", epochTime, finalLoss, finalAcc);
    }
}