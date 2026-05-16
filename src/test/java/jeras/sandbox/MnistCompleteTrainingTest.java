package jeras.sandbox;

import jeras.core.Tensor;
import jeras.layers.Dense;
import jeras.models.Sequential;
import jeras.utils.DataLoader;
import java.io.IOException;

public class MnistCompleteTrainingTest {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("==========================================================");
        System.out.println("🚀 Jeras v1.1 - 가상 스레드 셔플 파이프라인 다중 에포크 훈련 시작");
        System.out.println("==========================================================");

        // 데이터 스펙 및 하이퍼파라미터 세팅
        int totalSamples = 60_000;
        int batchSize = 100;
        int inputDim = 784;
        float learningRate = 0.0005f; // 안정적이면서 정밀한 그라디언트 하향을 위한 LR
        int epochs = 10;               // 🔥 총 5바퀴 지옥 훈련 세팅

        // 1. 모델 구성
        Sequential model = new Sequential(inputDim);
        Dense layer1 = new Dense(128, "relu");
        Dense layer2 = new Dense(10, "softmax");
        model.add(layer1);
        model.add(layer2);

        // 2. 가상 스레드 데이터 로더 초기화
        String imagesPath = "data/train-images.idx3-ubyte";
        String labelsPath = "data/train-labels.idx1-ubyte";

        DataLoader dataLoader = new DataLoader(batchSize, inputDim);
        System.out.printf("[DataLoader] 전담 가상 스레드가 백그라운드에서 총 %d 에포크 로드를 준비합니다...%n", epochs);
        dataLoader.startAsyncLoading(imagesPath, labelsPath, totalSamples, epochs);

        int totalBatches = totalSamples / batchSize;
        long startTime = System.nanoTime();

        System.out.println("[Engine] 연산 엔진 가동. 대규모 분산 비동기 큐잉 가속 루프 진입.\n");

        // 3. 다중 에포크 학습 루프 실행
        for (int epoch = 1; epoch <= epochs; epoch++) {
            long epochStartTime = System.nanoTime();
            float epochLoss = 0f;

            for (int b = 1; b <= totalBatches; b++) {
                // 가상 스레드가 큐에 예쁘게 셔플해서 채워둔 완벽한 배치를 순차 쟁취
                DataLoader.DataBatch batch = dataLoader.nextBatch();

                // 1) 순전파
                Tensor output = model.predict(batch.x());

                // 2) 오차 그라디언트 계산
                Tensor lossGradient = new Tensor(batchSize, 10);
                float batchLoss = 0f;
                for (int i = 0; i < batchSize * 10; i++) {
                    lossGradient.data[i] = output.data[i] - batch.y().data[i];
                    batchLoss += Math.abs(lossGradient.data[i]);
                }
                epochLoss += (batchLoss / batchSize);

                // 3) 역전파 가중치 학습 연산
                Tensor intermediateGrad = layer2.backward(layer1.forward(batch.x()), lossGradient, learningRate);
                layer1.backward(batch.x(), intermediateGrad, learningRate);

                // 200개 배치 단위로 내부 진행 모니터링
                if (b % 200 == 0) {
                    System.out.printf("   └ [Epoch %d] - [배치 %3d/%3d] 현재 배치 오차율: %.4f%n",
                            epoch, b, totalBatches, batchLoss / batchSize);
                }
            }

            long epochEndTime = System.nanoTime();
            System.out.printf("📢 [Epoch %d 완수] ➡️ 평균 오차율: %.4f | 소요 시간: %.2f 초%n",
                    epoch, epochLoss / totalBatches, (epochEndTime - epochStartTime) / 1_000_000_000.0);
        }

        long endTime = System.nanoTime();
        dataLoader.close();

        System.out.println("==========================================================");
        System.out.printf("🎯 지옥의 %d에포크 최종 훈련 종료! 총 소요 시간: %.2f 초%n", epochs, (endTime - startTime) / 1_000_000_000.0);

        // 4. 컴파일 에러 없는 가중치 추출 모델 저장
        String savePath = "data/jeras_mnist_model.txt";
        try {
            System.out.println("[Engine] 안전한 객체지향 캡슐화 가중치 추출 파이프라인 기동...");
            model.saveWeights(savePath);
        } catch (IOException e) {
            System.err.println("❌ 모델 가중치 파일 저장 중 실패: " + e.getMessage());
        }

        System.out.println("==========================================================");
    }
}