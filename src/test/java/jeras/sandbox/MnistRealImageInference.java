//package jeras.sandbox;
//
//import jeras.core.Tensor;
//import jeras.layers.Dense;
//import jeras.models.Sequential;
//
//import javax.imageio.ImageIO;
//import java.awt.*;
//import java.awt.image.BufferedImage;
//import java.io.File;
//
//public class MnistRealImageInference {
//    public static void main(String[] args) throws Exception {
//        System.out.println("🖼️ [Jeras v1.4] 스마트 전처리 엔진(Centering & Padding) 가동");
//
//        // 1. 모델 아키텍처 정의 (784 -> 256 -> 128 -> 10)
//        Dense layer1 = new Dense(512, "relu");
//        Dense layer2 = new Dense(256, "relu");
//        Dense layer3 = new Dense(128, "relu");
//        Dense layer4 = new Dense(10, "softmax");
//
//        Sequential model = new Sequential(784);
//        model.add(layer1);
//        model.add(layer2);
//        model.add(layer3);
//        model.add(layer4);
//
//        // 🌟 방금 구워진 딥 증강 가중치 파일 로드
//        String modelPath = "data/jeras_mnist_deep_augmented_model.txt";
//        model.loadWeights(modelPath);
//
//        System.out.println("\n🎯 --- 그림판 손글씨 10종 스마트 검증 시작 ---");
//
//        int correctCount = 0;
//        int totalCount = 10;
//
//        // ME0.png 부터 ME9.png 까지 순차 루프 돌기
//        for (int i = 0; i <= 9; i++) {
//            String filename = "ME" + i + ".png";
//            String imagePath = "data/" + filename;
//            File imageFile = new File(imagePath);
//
//            if (!imageFile.exists()) {
//                System.out.printf("⚠️ [파일 유실] %s 파일이 data/ 폴더에 없어 건너뜁니다.%n", filename);
//                totalCount--;
//                continue;
//            }
//
//            // 3. 업그레이드된 스마트 전처리 엔진 통과
//            float[] inputPixels = preprocessImage(imageFile);
//
//            // 4. 텐서 주입 및 예측
//            Tensor inputTensor = new Tensor(1, 784);
//            System.arraycopy(inputPixels, 0, inputTensor.data, 0, 784);
//            Tensor output = model.predict(inputTensor);
//
//            // 5. AI의 판정 인덱스 계산
//            int predictedDigit = 0;
//            float maxProb = output.data[0];
//            for (int o = 1; o < 10; o++) {
//                if (output.data[o] > maxProb) {
//                    maxProb = output.data[o];
//                    predictedDigit = o;
//                }
//            }
//
//            // 6. 정답 검증
//            int actualDigit = i;
//            boolean isCorrect = (predictedDigit == actualDigit);
//
//            if (isCorrect) {
//                correctCount++;
//            }
//
//            // 결과 개별 출력
//            System.out.printf("[%s] AI 판정: %d (신뢰도: %5.2f%%) | 실제 정답: %d -> %s%n",
//                    filename, predictedDigit, maxProb * 100f, actualDigit,
//                    isCorrect ? "정답! 🎉" : "오답.. 😭");
//        }
//
//        // 7. 최종 그림판 데이터셋 검증 리포트
//        System.out.println("----------------------------------------------------------------");
//        if (totalCount > 0) {
//            float customAccuracy = ((float) correctCount / totalCount) * 100f;
//            System.out.printf("📊 [스마트 검증 결과] 총 %d장 중 %d장 적중 | 최종 정확도: %.2f%%%n",
//                    totalCount, correctCount, customAccuracy);
//        }
//    }
//
//    /**
//     * 🛠️ [업그레이드 버전] 실제 이미지의 글자 영역을 자동 탐지하여
//     * MNIST 규격(정중앙 정렬 + 사방 여백 + 이진화)으로 완벽 보정하는 전처리 파이프라인
//     */
//    private static float[] preprocessImage(File file) throws Exception {
//        // 1. 원본 이미지 로드
//        BufferedImage src = ImageIO.read(file);
//        int w = src.getWidth();
//        int h = src.getHeight();
//
//        // 2. 글자가 있는 최소/최대 범위(Bounding Box) 찾기
//        int minX = w, minY = h, maxX = -1, maxY = -1;
//        boolean hasPixel = false;
//
//        for (int y = 0; y < h; y++) {
//            for (int x = 0; x < w; x++) {
//                int rgb = src.getRGB(x, y);
//                int r = (rgb >> 16) & 0xFF;
//
//                // 임계값 50을 넘는 밝은 픽셀(글자 부위) 감지
//                if (r > 50) {
//                    if (x < minX) minX = x;
//                    if (x > maxX) maxX = x;
//                    if (y < minY) minY = y;
//                    if (y > maxY) maxY = y;
//                    hasPixel = true;
//                }
//            }
//        }
//
//        // 만약 통째로 까만 빈 이미지라면 기본 처리로 탈출
//        if (!hasPixel) {
//            return new float[784];
//        }
//
//        // 3. 글자 알맹이 영역만 크롭(Crop)
//        int digitW = (maxX - minX) + 1;
//        int digitH = (maxY - minY) + 1;
//        BufferedImage cropped = src.getSubimage(minX, minY, digitW, digitH);
//
//        // 4. MNIST 교과서 규격(28x28)의 깨끗한 도화지 새로 준비
//        BufferedImage finalizedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
//        Graphics2D g2d = finalizedImage.createGraphics();
//
//        // 배경을 완전히 검은색으로 리셋
//        g2d.setColor(Color.BLACK);
//        g2d.fillRect(0, 0, 28, 28);
//
//        // 5. 글자의 비율을 유지하면서 20x20 크기 안에 쏙 들어가도록 리사이징 비율 계산
//        double scale = 20.0 / Math.max(digitW, digitH);
//        int transformW = (int) (digitW * scale);
//        int transformH = (int) (digitH * scale);
//
//        // 6. 28x28 도화지의 정확한 '정중앙' 좌표 계산 (여백 자동 배정)
//        int targetX = (28 - transformW) / 2;
//        int targetY = (28 - transformH) / 2;
//
//        // 렌더링 품질 힌트 세팅 (리사이징 시 글자 깨짐 방지 안티앨리어싱)
//        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
//
//        // 도화지 중앙에 글자 안착
//        g2d.drawImage(cropped, targetX, targetY, transformW, transformH, null);
//        g2d.dispose();
//
//        // 7. 최종 28x28 그리드를 돌며 0.0~1.0 배열로 직렬화 (이진화 필터링 포함)
//        float[] pixels = new float[784];
//        int idx = 0;
//        for (int y = 0; y < 28; y++) {
//            for (int x = 0; x < 28; x++) {
//                int rgb = finalizedImage.getRGB(x, y);
//                int r = (rgb >> 16) & 0xFF;
//
//                // 잔상을 지우고 글자 윤곽을 뚜렷하게 만들기 위한 이진화 가중치 조절
//                float val = r / 255.0f;
//                pixels[idx++] = val > 0.15f ? Math.min(1.0f, val * 1.2f) : 0.0f;
//            }
//        }
//
//        return pixels;
//    }
//}