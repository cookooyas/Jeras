package jeras.models;

import jeras.core.Tensor;
import jeras.layers.Layer;
import jeras.layers.Dense;
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

    public Tensor predict(Tensor input) {
        Tensor currentOutput = input;
        for (Layer layer : layers) {
            currentOutput = layer.forward(currentOutput);
        }
        return currentOutput;
    }

    // 🌟 Dense 클래스의 대소문자 가중치 배열 게터 명칭과 완벽 싱크 완료
    public void saveWeights(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (int i = 0; i < layers.size(); i++) {
                Layer genericLayer = layers.get(i);
                if (genericLayer instanceof Dense) {
                    Dense layer = (Dense) genericLayer;

                    writer.write("LAYER_" + i + "_WEIGHTS\n");
                    for (float w : layer.getWeights()) {
                        writer.write(w + ",");
                    }

                    writer.write("\nLAYER_" + i + "_BIAS\n");
                    for (float b : layer.getBias()) {
                        writer.write(b + ",");
                    }
                    writer.write("\n");
                }
            }
        }
        System.out.println("[Jeras Model] 💾 최종 가중치 추출 성공: " + filePath);
    }

    public void loadWeights(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int layerIdx = -1;
            Dense denseLayer = null;
            boolean isBiasMode = false;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                if (line.startsWith("LAYER_")) {
                    String[] parts = line.split("_");
                    layerIdx = Integer.parseInt(parts[1]);
                    isBiasMode = parts[2].equalsIgnoreCase("BIAS");

                    Layer genericLayer = layers.get(layerIdx);
                    if (genericLayer instanceof Dense) {
                        denseLayer = (Dense) genericLayer;
                    } else {
                        denseLayer = null;
                    }
                } else {
                    String[] tokens = line.split(",");
                    if (denseLayer != null && tokens.length > 0) {
                        float[] targetData = isBiasMode ? denseLayer.getBias() : denseLayer.getWeights();

                        for (int i = 0; i < Math.min(tokens.length, targetData.length); i++) {
                            targetData[i] = Float.parseFloat(tokens[i]);
                        }
                    }
                }
            }
        }
        System.out.println("[Jeras Model] 📂 가중치 복원 완료: " + filePath);
    }

    public List<Layer> getLayers() {
        return this.layers;
    }
}