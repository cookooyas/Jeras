package jeras.utils;

import jeras.core.Tensor;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

public class DataLoader implements AutoCloseable {
    private final BlockingQueue<DataBatch> dataQueue = new LinkedBlockingQueue<>(10);
    private final int batchSize;
    private final int inputDim;

    // 가상 스레드 풀을 멤버 변수로 관리함. 메서드 호출시 블라킹 방지
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();

    public DataLoader(int batchSize, int inputDim) {
        this.batchSize = batchSize;
        this.inputDim = inputDim;
    }

    public record DataBatch(Tensor x, Tensor y) {
    }

    public void startAsyncLoading(int totalSamples) {
        int numBatches = totalSamples / batchSize;

        for (int i = 0; i < numBatches; i++) {
            // 가상스레드 병목지점 해소 (연산엔진과 별도로 계속 동작하게 수정)
            executor.submit(() -> {
                try {
                    Thread.sleep((long) (Math.random() * 10)); // I/O 지연

                    Tensor batchX = new Tensor(batchSize, inputDim);
                    Tensor batchY = new Tensor(batchSize, 10);
                    batchX.initRandom();
                    batchY.initRandom();

                    dataQueue.put(new DataBatch(batchX, batchY));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
    }

    public DataBatch nextBatch() throws InterruptedException {
        return dataQueue.take();
    }

    @Override
    public void close() {
        executor.shutdown();
    }
}