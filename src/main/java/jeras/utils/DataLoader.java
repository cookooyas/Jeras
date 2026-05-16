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

public class DataLoader implements AutoCloseable {
    private final BlockingQueue<DataBatch> dataQueue = new LinkedBlockingQueue<>(20); // 버퍼 크기 확장
    private final int batchSize;
    private final int inputDim;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Random rand = new Random(42); // 셔플용 시드 고정

    public DataLoader(int batchSize, int inputDim) {
        this.batchSize = batchSize;
        this.inputDim = inputDim;
    }

    public record DataBatch(Tensor x, Tensor y) {}

    /**
     * 단 하나의 전담 가상 스레드가 백그라운드에서 안전하게 스트림을 읽으며
     * 다중 에포크와 데이터 셔플을 처리해 큐를 채웁니다.
     */
    public void startAsyncLoading(String imagesPath, String labelsPath, int totalSamples, int epochs) {
        executor.submit(() -> {
            try {
                int numBatches = totalSamples / batchSize;

                // 1. MNIST 데이터셋 전체를 가상 스레드 내부 로컬 버퍼에 원샷 적재 (하드디스크 병목 원천 차단)
                byte[] allImageBytes = new byte[totalSamples * inputDim];
                byte[] allLabelBytes = new byte[totalSamples];

                try (BufferedInputStream imgStream = new BufferedInputStream(new FileInputStream(imagesPath));
                     BufferedInputStream lblStream = new BufferedInputStream(new FileInputStream(labelsPath))) {
                    imgStream.skip(16); // 헤더 스킵
                    lblStream.skip(8);

                    int imgRead = imgStream.read(allImageBytes);
                    int lblRead = lblStream.read(allLabelBytes);
                    if (imgRead == -1 || lblRead == -1) return;
                }

                // 2. 인덱스 배열 준비
                int[] indices = new int[totalSamples];
                for (int i = 0; i < totalSamples; i++) {
                    indices[i] = i;
                }

                // 3. 다중 에포크 루프 수행
                for (int epoch = 1; epoch <= epochs; epoch++) {
                    // 에포크마다 무작위 데이터 뒤섞기 (Shuffle)
                    for (int i = indices.length - 1; i > 0; i--) {
                        int j = rand.nextInt(i + 1);
                        int temp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = temp;
                    }

                    // 4. 배치를 쪼개서 큐에 순차 주입 (FIFO 보장)
                    for (int bIdx = 0; bIdx < numBatches; bIdx++) {
                        Tensor batchX = new Tensor(batchSize, inputDim);
                        Tensor batchY = new Tensor(batchSize, 10);

                        for (int b = 0; b < batchSize; b++) {
                            int sampleIdx = indices[bIdx * batchSize + b];

                            // 이미지 데이터 변환
                            for (int p = 0; p < inputDim; p++) {
                                int pixel = allImageBytes[sampleIdx * inputDim + p] & 0xFF;
                                batchX.data[b * inputDim + p] = pixel / 255.0f;
                            }

                            // 정답 원-핫 인코딩 변환
                            int label = allLabelBytes[sampleIdx] & 0xFF;
                            batchY.data[b * 10 + label] = 1.0f;
                        }

                        // 메인 엔진이 가져갈 큐에 대기 주입 (큐가 차면 가상 스레드는 논블로킹 대기)
                        dataQueue.put(new DataBatch(batchX, batchY));
                    }
                }
            } catch (IOException | InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
    }

    public DataBatch nextBatch() throws InterruptedException {
        return dataQueue.take();
    }

    @Override
    public void close() {
        executor.shutdownNow();
    }
}