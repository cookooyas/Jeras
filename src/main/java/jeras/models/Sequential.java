package jeras.models;

import jeras.core.Tensor;
import jeras.layers.Layer;

import java.util.ArrayList;
import java.util.List;
import java.io.*;

public class Sequential {
    private final List<Layer> layers = new ArrayList<>();
    private int nextInputDim;

    public Sequential(int inputShape) {
        this.nextInputDim = inputShape;
    }

    public void add(Layer layer) {
        layer.initialize(this.nextInputDim);
        layers.add(layer);
        this.nextInputDim = layer.getOutputDim();
    }

    // 순전파
    public Tensor predict(Tensor input) {
        Tensor currentOutput = input;

        for (Layer layer : layers) {
            currentOutput = layer.forward(currentOutput);
        }

        return currentOutput;
    }

    // save
    public void saveWeights(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (int i = 0; i < layers.size(); i++) {
                Object genericLayer = layers.get(i);
                if (genericLayer instanceof jeras.layers.Dense) {
                    jeras.layers.Dense layer = (jeras.layers.Dense) genericLayer;

                    writer.write("LAYER_" + i + "_WEIGHTS\n");
                    for (float w : layer.getWeights().data) {
                        writer.write(w + ",");
                    }

                    writer.write("\nLAYER_" + i + "_BIAS\n");
                    for (float b : layer.getBias().data) {
                        writer.write(b + ",");
                    }
                    writer.write("\n");
                }
            }
        }
        System.out.println("[Jeras Model] 💾 최종 학습된 가중치 모델 추출 완료: " + filePath);
    }

    // load
    public void loadWeights(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int layerIdx = -1;
            jeras.layers.Dense denseLayer = null;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                if (line.startsWith("LAYER_")) {
                    String[] parts = line.split("_");
                    layerIdx = Integer.parseInt(parts[1]);

                    Object genericLayer = layers.get(layerIdx);
                    if (genericLayer instanceof jeras.layers.Dense) {
                        denseLayer = (jeras.layers.Dense) genericLayer;
                    } else {
                        denseLayer = null;
                    }
                } else {
                    String[] tokens = line.split(",");
                    if (denseLayer != null && tokens.length > 0 && line.contains(".")) {

                        boolean isBias = (denseLayer.getBias().data.length == tokens.length);
                        float[] targetData = isBias ? denseLayer.getBias().data : denseLayer.getWeights().data;

                        for (int i = 0; i < Math.min(tokens.length, targetData.length); i++) {
                            targetData[i] = Float.parseFloat(tokens[i]);
                        }
                    }
                }
            }
        }
        System.out.println("[Jeras Model] 📂 가중치 모델 로드 및 뇌 세포 복원 완료: " + filePath);
    }
}