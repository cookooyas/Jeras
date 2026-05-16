package jeras.sandbox;

import jeras.core.Tensor;
import jeras.layers.Dense;
import jeras.models.Sequential;
import jeras.utils.DataLoader;

public class MnistCompleteTrainingTest {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("==========================================================");
        System.out.println("🚀 Jeras v1.0 - 순수 Java 25 기반 진짜 MNIST 인공지능 학습 시작");
        System.out.println("==========================================================");

        // 데이터 스펙 세팅
        int totalSamples = 60_000; // MNIST 전체 훈련 데이터 개수
        int batchSize = 100;
        int inputDim = 784;        // 28x28 이미지
        float learningRate = 0.0001f;

        // 1. 모델 구성 (Keras 스타일 감성)
        Sequential model = new Sequential(inputDim);
        Dense layer1 = new Dense(128, "relu");
        Dense layer2 = new Dense(10, "softmax"); // 최종 0~9 분류
        model.add(layer1);
        model.add(layer2);

        // 2. 가상 스레드 데이터 로더 초기화
        String imagesPath = "data/train-images.idx3-ubyte";
        String labelsPath = "data/train-labels.idx1-ubyte";

        DataLoader dataLoader = new DataLoader(batchSize, inputDim);
        System.out.println("[DataLoader] 가상 스레드 군단이 백그라운드에서 진짜 고용량 이미지 파일을 읽기 시작합니다...");
        dataLoader.startAsyncLoading(imagesPath, labelsPath, totalSamples);

        int totalBatches = totalSamples / batchSize;
        long startTime = System.nanoTime();

        System.out.println("[Engine] 연산 엔진 가동. 총 60,000장의 이미지 대규모 가속 학습 루프 진입.\n");

        // 3. 60,000장 이미지 학습 루프 실행
        for (int b = 1; b <= totalBatches; b++) {
            // 가상 스레드가 디스크에서 실시간으로 퍼다 나른 진짜 이미지/정답 세트 쟁취
            DataLoader.DataBatch batch = dataLoader.nextBatch();

            // 1) 순전파 (현재 뇌 상태로 이미지 예측)
            Tensor output = model.predict(batch.x());

            // 2) 오차 그라디언트 계산 (Loss Gradient: 예측값 - 실제 정답)
            Tensor lossGradient = new Tensor(batchSize, 10);
            float batchLoss = 0f;
            for (int i = 0; i < batchSize * 10; i++) {
                lossGradient.data[i] = output.data[i] - batch.y().data[i];
                batchLoss += Math.abs(lossGradient.data[i]); // 모니터링용 오차 누적
            }

            // 3) 역전파 학습 연산 (거꾸로 올라가며 가중치 수정!)
            // 최종 출력층부터 뒤에서 앞으로 오차를 전파하며 깎아 나갑니다.
            Tensor intermediateGrad = layer2.backward(layer1.forward(batch.x()), lossGradient, learningRate);
            layer1.backward(batch.x(), intermediateGrad, learningRate);

            // 100개 배치 단위로 학습 진행 상황 모니터링 (오차가 줄어드는지 확인!)
            if (b % 100 == 0 || b == totalBatches) {
                System.out.printf("   [배치 %3d/%3d] 훈련 완료 ➡️ 현재 배치 평균 오차율: %.4f%n",
                        b, totalBatches, batchLoss / batchSize);
            }
        }

        long endTime = System.nanoTime();
        dataLoader.close();

        System.out.println("==========================================================");
        System.out.printf("🎯 학습 종료! 60,000장 전량 학습 소요 시간: %.2f 초%n", (endTime - startTime) / 1_000_000_000.0);
        System.out.println("💡 배치가 진행될수록 오차율이 뚝뚝 떨어졌다면 AI가 진짜 손글씨를 배운 것입니다!");
        System.out.println("==========================================================");
    }
}