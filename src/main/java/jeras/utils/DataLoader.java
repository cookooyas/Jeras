package jeras.utils;

import jeras.core.Tensor;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

public class DataLoader implements AutoCloseable {
    private final BlockingQueue<DataBatch> dataQueue = new LinkedBlockingQueue<>(10);
    private final int batchSize;
    private final int inputDim;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();

    public DataLoader(int batchSize, int inputDim) {
        this.batchSize = batchSize;
        this.inputDim = inputDim;
    }

    public record DataBatch(Tensor x, Tensor y) {}

    public void startAsyncLoading(String imagesPath, String labelsPath, int totalSamples) {
        int numBatches = totalSamples / batchSize;

        for (int i = 0; i < numBatches; i++) {
            final int batchIdx = i;
            executor.submit(() -> {
                try (BufferedInputStream imgStream = new BufferedInputStream(new FileInputStream(imagesPath));
                     BufferedInputStream lblStream = new BufferedInputStream(new FileInputStream(labelsPath))) {

                    long imgSkip = 16 + (long) batchIdx * batchSize * inputDim;
                    long lblSkip = 8 + (long) batchIdx * batchSize;
                    imgStream.skip(imgSkip);
                    lblStream.skip(lblSkip);

                    Tensor batchX = new Tensor(batchSize, inputDim);
                    Tensor batchY = new Tensor(batchSize, 10); // 정답 원-핫 인코딩용 (0~9)

                    for (int b = 0; b < batchSize; b++) {
                        for (int p = 0; p < inputDim; p++) {
                            int pixel = imgStream.read();
                            batchX.data[b * inputDim + p] = pixel / 255.0f;
                        }
                    }

                    for (int b = 0; b < batchSize; b++) {
                        int label = lblStream.read();
                        batchY.data[b * 10 + label] = 1.0f;
                    }

                    dataQueue.put(new DataBatch(batchX, batchY));

                } catch (IOException | InterruptedException e) {
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