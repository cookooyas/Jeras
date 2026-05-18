//package jeras.sandbox;
//
//import jeras.core.Tensor;
//import jeras.layers.Dense;
//import jeras.models.Sequential;
//import jeras.utils.DataLoader;
//
//public class MnistCompleteTrainingTest {
//    public static void main(String[] args) throws Exception {
//        System.out.println("🔥 [Jeras v1.4] 가속 연산 + 오버플로우 방어막 지옥 훈련 시작");
//
//        int totalSamples = 60_000;
//        int batchSize = 100;
//        float learningRate = 0.0002f; // 수렴을 위해 약간 하향 조정
//        int epochs = 10;
//
//        // 🌟 3층 구조 맵핑 (784 -> 256 -> 128 -> 10)
//        Dense layer1 = new Dense(784, 256, "relu");
//        Dense layer2 = new Dense(256, 128, "relu");
//        Dense layer3 = new Dense(128, 10, "softmax");
//
//        Sequential model = new Sequential(784);
//        model.add(layer1);
//        model.add(layer2);
//        model.add(layer3);
//
//        DataLoader dataLoader = new DataLoader(batchSize, 784);
//        dataLoader.startAsyncLoading("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", totalSamples, epochs);
//
//        for (int epoch = 1; epoch <= epochs; epoch++) {
//            float epochLoss = 0f;
//            for (int b = 1; b <= totalSamples / batchSize; b++) {
//                DataLoader.DataBatch batch = dataLoader.nextBatch();
//                Tensor output = model.predict(batch.x());
//
//                Tensor lossGrad = new Tensor(batchSize, 10);
//                float bLoss = 0;
//                for (int i = 0; i < batchSize * 10; i++) {
//                    lossGrad.data[i] = output.data[i] - batch.y().data[i];
//                    bLoss += Math.abs(lossGrad.data[i]);
//                }
//                epochLoss += (bLoss / batchSize);
//
//                // 상속 규격과 전파 순서 완벽 검증 완료된 체인
//                Tensor g3 = layer3.backward(layer2.forward(layer1.forward(batch.x())), lossGrad, learningRate);
//                Tensor g2 = layer2.backward(layer1.forward(batch.x()), g3, learningRate);
//                layer1.backward(batch.x(), g2, learningRate);
//            }
//            System.out.printf("[Epoch %d] 평균 오차율: %.4f%n", epoch, epochLoss / (totalSamples / batchSize));
//        }
//        model.saveWeights("data/jeras_mnist_model.txt");
//        dataLoader.close();
//    }
//}