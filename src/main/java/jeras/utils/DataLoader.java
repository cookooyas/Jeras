package jeras.utils;

import jeras.core.Tensor;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * 🚚 [Jeras Utility] 가상 스레드 기반 초고속 비동기 데이터 파이프라인 로더
 * TensorFlow의 tf.data.Dataset 프리페치(Prefetch) 메커니즘을 자바 가상 스레드로 완벽히 재현했습니다.
 * 픽셀 가공 로직을 분리하고, 순수 배치 공급 및 셔플 셔틀 역할에 집중합니다.
 */
public class DataLoader implements AutoCloseable {
    private final BlockingQueue<DataBatch> dataQueue = new LinkedBlockingQueue<>(20);
    private final int batchSize;
    private final int inputDim;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Random rand = new Random(42);

    // 🌟 앞서 설계한 명품 Augmentation 엔진을 주입받아 유연성 확보
    private final Augmentation augmenter;

    public record DataBatch(Tensor x, Tensor y) {}

    /**
     * 기본 로더 생성자 (데이터 증강 없음)
     */
    public DataLoader(int batchSize, int inputDim) {
        this(batchSize, inputDim, null);
    }

    /**
     * 데이터 증강 엔진을 주입받는 Keras 스타일 생성자
     */
    public DataLoader(int batchSize, int inputDim, Augmentation augmenter) {
        this.batchSize = batchSize;
        this.inputDim = inputDim;
        this.augmenter = augmenter;
    }

    /**
     * ⚡ [Background Engine] 백그라운드 전담 가상 스레드를 가동하여
     * 디스크 I/O 병목을 제로로 만들고 큐에 배치를 미리 탑재(Prefetch)합니다.
     */
    public void startAsyncLoading(String imagesPath, String labelsPath, int totalSamples, int epochs) {
        executor.submit(() -> {
            try {
                int numBatches = totalSamples / batchSize;

                // 1. MNIST 원본 바이너리 데이터를 메모리에 원샷 버퍼 적재 (I/O 병목 제거)
                byte[] allImageBytes = new byte[totalSamples * inputDim];
                byte[] allLabelBytes = new byte[totalSamples];

                try (BufferedInputStream imgStream = new BufferedInputStream(new FileInputStream(imagesPath));
                     BufferedInputStream lblStream = new BufferedInputStream(new FileInputStream(labelsPath))) {
                    imgStream.skip(16); // MNIST IDX 헤더 건너뛰기
                    lblStream.skip(8);

                    imgStream.read(allImageBytes);
                    lblStream.read(allLabelBytes);
                }

                // 2. 무작위 셔플을 위한 인덱스 평면 매핑
                int[] indices = new int[totalSamples];
                for (int i = 0; i < totalSamples; i++) indices[i] = i;

                // 3. 에포크 스케줄링 가동
                for (int epoch = 1; epoch <= epochs; epoch++) {
                    // 🎲 Keras의 shuffle=True 옵션 구현 (에포크마다 인덱스를 완전히 뒤섞음)
                    for (int i = indices.length - 1; i > 0; i--) {
                        int j = rand.nextInt(i + 1);
                        int temp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = temp;
                    }

                    // 4. 미니배치 텐서 빌드 및 큐 적재 루프
                    for (int bIdx = 0; bIdx < numBatches; bIdx++) {
                        Tensor batchX = new Tensor(batchSize, inputDim);
                        Tensor batchY = new Tensor(batchSize, 10); // 10개 카테고리 (0~9)

                        for (int b = 0; b < batchSize; b++) {
                            int sampleIdx = indices[bIdx * batchSize + b];

                            // 1장 분량의 원본 픽셀 정규화 [0.0 ~ 1.0] 추출
                            float[] singleImage = new float[inputDim];
                            int srcSampleOffset = sampleIdx * inputDim;
                            for (int p = 0; p < inputDim; p++) {
                                singleImage[p] = (allImageBytes[srcSampleOffset + p] & 0xFF) / 255.0f;
                            }

                            // 🌟 [추상화 결합 해제] 만약 증강 엔진이 주입되었다면 기하학 변형 가동!
                            if (this.augmenter != null) {
                                singleImage = this.augmenter.augment(singleImage);
                            }

                            // 최종 미니배치 텐서 버퍼 배열에 바인딩
                            System.arraycopy(singleImage, 0, batchX.data, b * inputDim, inputDim);

                            // 정답 정수 라벨 추출 후 원핫 인코딩(One-Hot Encoding) 자동 전개
                            int label = allLabelBytes[sampleIdx] & 0xFF;
                            batchY.data[b * 10 + label] = 1.0f;
                        }

                        // 큐가 가득 차면 가상 스레드는 블로킹되어 대기 (메모리 폭주 방지)
                        dataQueue.put(new DataBatch(batchX, batchY));
                    }
                }
            } catch (IOException | InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
    }

    /**
     * 🏁 훈련 엔진이 큐에서 준비된 배치를 소모할 때 호출하는 API (FIFO)
     */
    public DataBatch nextBatch() throws InterruptedException {
        return dataQueue.take();
    }

    @Override
    public void close() {
        executor.shutdownNow();
    }
}