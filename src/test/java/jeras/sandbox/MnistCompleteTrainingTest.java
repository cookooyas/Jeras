package jeras.sandbox;

import jeras.core.Tensor;
import jeras.layers.Dense;
import jeras.models.Sequential;
import jeras.utils.DataLoader;

public class MnistCompleteTrainingTest {
    public static void main(String[] args) throws Exception {
        System.out.println("🔥 [Jeras v1.4] 데이터 증강 + 다층 레이어(3층) 지옥 훈련 시작");

        int totalSamples = 60_000;
        int batchSize = 100;
        float learningRate = 0.0005f;
        int epochs = 10; // 🌟 증강이 들어가면 학습이 더 오래 필요하므로 10에포크 권장

        // 🌟 3층 구조로 확장 (Capacity 증가)
        Sequential model = new Sequential(784);
        model.add(new Dense(256, "relu"));
        model.add(new Dense(128, "relu"));
        model.add(new Dense(10, "softmax"));

        DataLoader dataLoader = new DataLoader(batchSize, 784);
        dataLoader.startAsyncLoading("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", totalSamples, epochs);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            float epochLoss = 0f;
            for (int b = 1; b <= totalSamples / batchSize; b++) {
                DataLoader.DataBatch batch = dataLoader.nextBatch();
                Tensor output = model.predict(batch.x());

                Tensor lossGrad = new Tensor(batchSize, 10);
                float bLoss = 0;
                for (int i = 0; i < batchSize * 10; i++) {
                    lossGrad.data[i] = output.data[i] - batch.y().data[i];
                    bLoss += Math.abs(lossGrad.data[i]);
                }
                epochLoss += (bLoss / batchSize);

                // 역전파 (Chain Rule)
                Tensor g3 = ((Dense)model.getLayers().get(2)).backward(((Dense)model.getLayers().get(1)).forward(((Dense)model.getLayers().get(0)).forward(batch.x())), lossGrad, learningRate);
                Tensor g2 = ((Dense)model.getLayers().get(1)).backward(((Dense)model.getLayers().get(0)).forward(batch.x()), g3, learningRate);
                ((Dense)model.getLayers().get(0)).backward(batch.x(), g2, learningRate);
            }
            System.out.printf("[Epoch %d] 평균 오차율: %.4f%n", epoch, epochLoss / (totalSamples / batchSize));
        }
        model.saveWeights("data/jeras_mnist_model.txt");
        dataLoader.close();
    }
}