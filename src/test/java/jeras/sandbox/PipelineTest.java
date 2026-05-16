package jeras.sandbox;

import jeras.core.Tensor;
import jeras.layers.Dense;
import jeras.models.Sequential;
import jeras.utils.DataLoader;

public class PipelineTest {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("==================================================");
        System.out.println("   Jeras 파이프라인 테스트: 가상 스레드 DataLoader + Vector API  ");
        System.out.println("==================================================");

        int totalSamples = 10_000; // 총 1만 개의 이미지 데이터 처리 시뮬레이션
        int batchSize = 100;       // 한 번에 100개씩 묶어서 연산
        int inputDim = 784;

        // 1. 모델 빌드
        Sequential model = new Sequential(inputDim);
        model.add(new Dense(128, "relu"));
        model.add(new Dense(64, "relu"));
        model.add(new Dense(10, "softmax"));

        // 2. 가상 스레드 기반 DataLoader 기동
        DataLoader dataLoader = new DataLoader(batchSize, inputDim);

        System.out.println("[DataLoader] 백그라운드 가상 스레드 가동... 비동기 파일 로딩 시작");
        long startPipeline = System.nanoTime();

        // 백그라운드에서 가상 스레드들이 수십 개의 배치를 동시에 읽어오기 시작합니다.
        dataLoader.startAsyncLoading(totalSamples);

        int totalBatches = totalSamples / batchSize;
        System.out.println("[Engine] 연산 엔진 가동. 큐에서 데이터를 가져와 순전파 연산을 연속 수행합니다.\n");

        // 3. 메인 학습 루프 (연산 엔진 스레드)
        for (int b = 1; b <= totalBatches; b++) {
            // 가상 스레드가 디스크에서 읽어와 큐에 채워둔 따끈따끈한 데이터를 즉시 가져옵니다 (I/O 대기시간 0에 수렴)
            DataLoader.DataBatch batch = dataLoader.nextBatch();

            // 가져온 데이터를 Vector API 엔진으로 바로 밀어 넣고 계산!
            Tensor output = model.predict(batch.x());

            if (b % 20 == 0 || b == totalBatches) {
                System.out.printf("   [진행률: %3d/%3d 배치] 연산 완료 (현재 배치 결과 텐서 크기: %dx%d)%n",
                        b, totalBatches, output.rows, output.cols);
            }
        }

        long endPipeline = System.nanoTime();
        double durationSec = (endPipeline - startPipeline) / 1_000_000_000.0;

        System.out.println("==================================================");
        System.out.printf("🎯 전체 10,000개 데이터 [로딩 + 파일 I/O 대기 + 행렬 연산] 총 소요 시간: %.2f 초%n", durationSec);
        System.out.println("==================================================");
    }
}