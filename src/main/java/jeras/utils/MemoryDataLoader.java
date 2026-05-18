package jeras.utils;

import jeras.core.Tensor;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MemoryDataLoader {
    private final int batchSize;
    private final int inputDim;
    private final int totalSamples;

    private Tensor allImages;
    private Tensor allLabels;
    private int currentBatchIdx = 0;

    public MemoryDataLoader(int batchSize, int inputDim, int totalSamples) {
        this.batchSize = batchSize;
        this.inputDim = inputDim;
        this.totalSamples = totalSamples;
    }

    public void loadAllToMemory(String imagesPath, String labelsPath) throws IOException {
        System.out.println("[MemoryDataLoader] 실제 MNIST 파일 전체를 메모리에 로딩 중...");

        this.allImages = new Tensor(totalSamples, inputDim);
        this.allLabels = new Tensor(totalSamples, 10);

        try (BufferedInputStream imgStream = new BufferedInputStream(new FileInputStream(imagesPath));
             BufferedInputStream lblStream = new BufferedInputStream(new FileInputStream(labelsPath))) {

            imgStream.skip(16);
            lblStream.skip(8);

            for (int i = 0; i < totalSamples; i++) {
                for (int p = 0; p < inputDim; p++) {
                    int pixel = imgStream.read();
                    allImages.data[i * inputDim + p] = pixel / 255.0f;
                }
                int label = lblStream.read();
                allLabels.data[i * 10 + label] = 1.0f;
            }
        }
        System.out.println("[MemoryDataLoader] 로딩 완료! 60,000장 전량 메모리 적재 성공.");
    }

    public DataLoader.DataBatch nextBatch() {
        Tensor batchX = new Tensor(batchSize, inputDim);
        Tensor batchY = new Tensor(batchSize, 10);

        int startRow = currentBatchIdx * batchSize;

        System.arraycopy(allImages.data, startRow * inputDim, batchX.data, 0, batchSize * inputDim);
        System.arraycopy(allLabels.data, startRow * 10, batchY.data, 0, batchSize * 10);

        currentBatchIdx++;
        return new DataLoader.DataBatch(batchX, batchY);
    }
}