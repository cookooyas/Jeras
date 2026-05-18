//package jeras.sandbox;
//
//import jeras.core.Tensor;
//import jeras.layers.Dense;
//import jeras.models.Sequential;
//import jeras.utils.DataLoader;
//import jeras.utils.Augmentation;
//
//public class MnistAugmentedTrainingTest {
//    public static void main(String[] args) throws Exception {
//        System.out.println("🚀 [Jeras v1.4] 4층 신경망(Deep MLP) + 실시간 데이터 증강 마스터 훈련 가동");
//
//        int totalSamples = 60_000;
//        int batchSize = 100;
//        float learningRate = 0.00005f; // 깊은 레이어의 안정적 학습을 위한 최적 학습률
//        int epochs = 15;              // 약 7~8분 내 완주 가능한 실전 타겟 에포크 스케일
//
//        // 🌟 [레이어 확장 설계] 784 -> 512 -> 256 -> 128 -> 10
//        // 가중치 오버플로우 방어 및 그라디언트 소실 차단을 위해 내부적으로 He 초기화와 ReLU가 정상 작동해야 합니다.
//        Dense layer1 = new Dense(784, 512, "relu");   // 고차원 픽셀 특징 다량 추출
//        Dense layer2 = new Dense(512, 256, "relu");   // 추상적 선/면 형태 조립 1단계
//        Dense layer3 = new Dense(256, 128, "relu");   // 추상적 구조 조립 2단계
//        Dense layer4 = new Dense(128, 10, "softmax"); // 최종 확률 분포 연산층
//
//        Sequential model = new Sequential(784);
//        model.add(layer1);
//        model.add(layer2);
//        model.add(layer3);
//        model.add(layer4);
//
//        // 비동기 고속 데이터 로더 구동
//        DataLoader dataLoader = new DataLoader(batchSize, 784);
//        dataLoader.startAsyncLoading("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", totalSamples, epochs);
//
//        System.out.println("🏋️ 지옥의 35에포크 연쇄 증강 학습 파이프라인을 시작합니다...");
//        long startTime = System.currentTimeMillis();
//
//        for (int epoch = 1; epoch <= epochs; epoch++) {
//            float epochLoss = 0f;
//            int numBatches = totalSamples / batchSize;
//
//            for (int b = 1; b <= numBatches; b++) {
//                DataLoader.DataBatch batch = dataLoader.nextBatch();
//
//                // 🌟 [실시간 데이터 괴롭히기] 배치 내부의 모든 이미지를 무작위 회전/밀기 변환
//                Tensor augmentedX = new Tensor(batchSize, 784);
//                for (int s = 0; s < batchSize; s++) {
//                    float[] singleImage = new float[784];
//                    System.arraycopy(batch.x().data, s * 784, singleImage, 0, 784);
//
//                    // 무작위 증강 적용 (Rotation, Shift 변환 연산)
//                    float[] augmentedImage = Augmentation.augment(singleImage);
//                    System.arraycopy(augmentedImage, 0, augmentedX.data, s * 784, 784);
//                }
//
//                // 1. 순전파(Forward) 연산 체인
//                Tensor out1 = layer1.forward(augmentedX);
//                Tensor out2 = layer2.forward(out1);
//                Tensor out3 = layer3.forward(out2);
//                Tensor output = layer4.forward(out3);
//
//                // 2. 오차율(Loss Gradient) 측정 및 스케일 제어
//                Tensor lossGrad = new Tensor(batchSize, 10);
//                float bLoss = 0;
//                for (int i = 0; i < batchSize * 10; i++) {
//                    lossGrad.data[i] = output.data[i] - batch.y().data[i];
//                    bLoss += Math.abs(lossGrad.data[i]);
//                }
//                epochLoss += (bLoss / batchSize);
//
//                // 3. 🌟 역전파(Backward) 가중치 업데이트 체인 연장
//                Tensor g4 = layer4.backward(out3, lossGrad, learningRate);
//                Tensor g3 = layer3.backward(out2, g4, learningRate);
//                Tensor g2 = layer2.backward(out1, g3, learningRate);
//                layer1.backward(augmentedX, g2, learningRate);
//            }
//
//            // 5에포크 단위로 로그가 한눈에 보이도록 가시성 개선
//            if (epoch == 1 || epoch % 5 == 0 || epoch == epochs) {
//                System.out.printf("🎯 [Epoch %2d/%2d] 뇌세포 평균 오차율(Loss): %.4f%n",
//                        epoch, epochs, epochLoss / numBatches);
//            }
//        }
//
//        long endTime = System.currentTimeMillis();
//        System.out.printf("%n🏁 35에포크 지옥 훈련 완료! 총 소요 시간: %.2f초%n", (endTime - startTime) / 1000.0f);
//
//        // 🌟 실전용 마스터 가중치 파일로 완전히 격리 분리하여 안전 저장
//        String masterModelPath = "data/jeras_mnist_deep_augmented_model.txt";
//        model.saveWeights(masterModelPath);
//        System.out.println("💾 마스터 가중치가 안전하게 빌드되었습니다: " + masterModelPath);
//
//        dataLoader.close();
//    }
//}