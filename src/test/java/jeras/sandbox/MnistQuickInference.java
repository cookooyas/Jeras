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

    // 💡 테스트할 원본 이미지 경로와 가중치 파일 경로
    private static final String IMAGE_PATH = "data/my_number.png";
    private static final String MODEL_PATH = "data/jeras_mnist_model.txt";
    private static final String DEBUG_OUT_PATH = "data/debug_processed.png"; // 🌟 전처리 후 저장될 파일 경로

    public static void main(String[] args) throws Exception {
        System.out.println("==========================================================");
        System.out.println("🔮 Jeras v1.2 - 이미지 전처리 시각화 및 디버그 인퍼런스");
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

        // 3. 자바 가속 스케일링으로 28x28 Grayscale 버퍼 생성
        BufferedImage resizedImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resizedImg.createGraphics();
        g.drawImage(rawImg.getScaledInstance(28, 28, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();

        // 4. [전처리 피드백 및 두께 보정] 데이터 변환 및 임시 그리드 적재
        Tensor inputTensor = new Tensor(1, 784);
        float[][] tempGrid = new float[28][28];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int brightness = resizedImg.getRGB(x, y) & 0xFF;
                // 배경 흰색(255) -> 0.0(검은색), 글씨 검은색(0) -> 1.0(흰색) 반전
                tempGrid[y][x] = (255 - brightness) / 255.0f;
            }
        }

        // 5. 선 얇아짐 방지를 위한 팽창(Dilation) 보정 알고리즘 적용
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float maxVal = tempGrid[y][x];
                // 주변 3x3 이웃을 뒤져서 가장 진한 값을 흡수
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
                            if (tempGrid[ny][nx] > maxVal) {
                                maxVal = tempGrid[ny][nx];
                            }
                        }
                    }
                }
                // 노이즈 차단 및 밝기 부스트
                float finalPixel = maxVal > 0.15f ? Math.min(1.0f, maxVal * 1.8f) : 0.0f;
                inputTensor.data[y * 28 + x] = finalPixel;
            }
        }
        System.out.println("[ImageProcessor] 📷 원본 이미지 해상도 압축 및 784차원 텐서 변환 완료.");

        // 6. 🌟 [핵심 요구사항] AI가 바라보는 28x28 상태를 실제 이미지 파일로 저장하기
        BufferedImage debugImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float val = inputTensor.data[y * 28 + x]; // 0.0 ~ 1.0
                int pixelValue = (int) (val * 255);       // 0 ~ 255 (검은 배경에 흰 글씨 형태)

                // TYPE_BYTE_GRAY 포맷에 맞게 8비트 RGB 비트마스킹 주입
                int rgb = (pixelValue << 16) | (pixelValue << 8) | pixelValue;
                debugImg.setRGB(x, y, rgb);
            }
        }
        File debugFile = new File(DEBUG_OUT_PATH);
        ImageIO.write(debugImg, "png", debugFile);
        System.out.println("[DebugSystem] 💾 AI가 본 스냅샷 저장 완료: " + debugFile.getAbsolutePath());

        // 7. 👀 콘솔창에 텍스트 그래픽으로 실시간 매트릭스 시각화 출력
        System.out.println("\n🖥️ [콘솔 시각화 - AI 넷막에 맺힌 28x28 실제 상(像)]");
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float val = inputTensor.data[y * 28 + x];
                if (val > 0.6f) System.out.print("■");      // 매우 진함
                else if (val > 0.15f) System.out.print("▨"); // 경계선 회색조
                else System.out.print(" ");                 // 여백
            }
            System.out.println();
        }
        System.out.println();

        // 8. 가속 엔진 추론 가동
        long startTime = System.nanoTime();
        Tensor prediction = model.predict(inputTensor);
        long endTime = System.nanoTime();

        // 9. 확률 분포 분석 및 최종 인덱스 매칭
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