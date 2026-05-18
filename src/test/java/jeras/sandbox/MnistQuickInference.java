package jeras.sandbox;

import jeras.core.Tensor;
import jeras.layers.Dense;
import jeras.models.Sequential;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

public class MnistQuickInference {

    private static final String IMAGE_PATH = "data/MNIST0.png";
    private static final String MODEL_PATH = "data/jeras_mnist_model.txt";
    private static final String DEBUG_OUT_PATH = "data/debug_processed.png";

    public static void main(String[] args) throws Exception {
        System.out.println("==========================================================");
        System.out.println("🔮 Jeras v1.3 - 정석 전처리 매핑 및 디버그 인퍼런스");
        System.out.println("==========================================================");

        // 1. 모델 프레임 구축 및 가중치 이식
        Sequential model = new Sequential(784);
        Dense layer1 = new Dense(128, "relu");
        Dense layer2 = new Dense(10, "softmax");
        model.add(layer1);
        model.add(layer2);
        model.loadWeights(MODEL_PATH);

        // 2. 원본 이미지 파일 확인 및 로드
        File file = new File(IMAGE_PATH);
        if (!file.exists()) {
            System.err.println("❌ 지정한 경로에 이미지 파일이 없습니다: " + file.getAbsolutePath());
            return;
        }
        BufferedImage rawImg = ImageIO.read(file);

        // 3. 자바 고품질 스케일링으로 28x28 Grayscale 버퍼 생성
        BufferedImage resizedImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resizedImg.createGraphics();
        // SCALE_SMOOTH를 사용하여 글자가 깨지거나 떡지는 것을 방지합니다.
        g.drawImage(rawImg.getScaledInstance(28, 28, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();

        // 4. [정석 전처리] 과도한 부스트나 팽창 없이 MNIST 표준 스타일로 순수 변환
        Tensor inputTensor = new Tensor(1, 784);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int brightness = resizedImg.getRGB(x, y) & 0xFF;

                // 배경 흰색(255) -> 0.0(검은색), 글씨 검은색(0) -> 1.0(흰색)
                float normalized = (255 - brightness) / 255.0f;

                // 💡 미세한 배경 노이즈(예: 스캔이나 캡처 시 생기는 미세한 흔들림)만 커트라인(0.1)으로 날려줍니다.
                if (normalized < 0.1f) {
                    normalized = 0.0f;
                }

                inputTensor.data[y * 28 + x] = normalized;
            }
        }
        System.out.println("[ImageProcessor] 📷 원본 이미지의 기하학적 곡선을 그대로 유지하며 텐서 변환 완료.");

        // 5. AI가 바라보는 28x28 상태를 실제 이미지 파일로 저장하기 (검증용)
        BufferedImage debugImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float val = inputTensor.data[y * 28 + x];
                int pixelValue = (int) (val * 255);
                int rgb = (pixelValue << 16) | (pixelValue << 8) | pixelValue;
                debugImg.setRGB(x, y, rgb);
            }
        }
        File debugFile = new File(DEBUG_OUT_PATH);
        ImageIO.write(debugImg, "png", debugFile);
        System.out.println("[DebugSystem] 💾 정석 전처리 스냅샷 재저장 완료: " + debugFile.getAbsolutePath());

        // 6. 👀 콘솔창에 텍스트 그래픽으로 실시간 매트릭스 시각화 출력
        System.out.println("\n🖥️ [콘솔 시각화 - 떡진 현상이 해결된 28x28 상(像)]");
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float val = inputTensor.data[y * 28 + x];
                if (val > 0.5f) System.out.print("■");
                else if (val > 0.1f) System.out.print("▨");
                else System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println();

        // 7. 가속 엔진 추론 가동
        long startTime = System.nanoTime();
        Tensor prediction = model.predict(inputTensor);
        long endTime = System.nanoTime();

        // 8. 확률 분포 분석 및 최종 인덱스 매칭
        int predictedDigit = -1;
        float maxConfidence = -1f;
        for (int i = 0; i < 10; i++) {
            if (prediction.data[i] > maxConfidence) {
                maxConfidence = prediction.data[i];
                predictedDigit = i;
            }
        }

        System.out.println("==========================================================");
        System.out.printf("🎯 AI 최종 판독 결과: [%d] %n", predictedDigit);
        System.out.printf("📊 판독 확신도 (Confidence): %.2f%%%n", maxConfidence * 100);
        System.out.printf("⏱️ 순수 추론 연산 타임: %.4f ms%n", (endTime - startTime) / 1_000_000.0);
        System.out.println("🔢 전체 클래스별 확률 분포: " + Arrays.toString(prediction.data));
        System.out.println("==========================================================");
    }
}