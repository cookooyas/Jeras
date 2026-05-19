package jeras;

import jeras.models.Sequential;
import jeras.layers.Dense;
import jeras.utils.Augmentation;
import jeras.utils.DataLoader;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * 🎛️ [Jeras 마스터 컨트롤러]
 * 상단의 플래그와 경로를 제어하여 학습(Train) 또는 커스텀 이미지 추론(Inference)을 원샷 구동합니다.
 */
public class JerasExecutionMaster {
    // 📂 공통 환경 설정 데이터 경로 (MNIST 바이너리 파일 경로)
    private static final String TRAIN_IMAGES = "./data/train-images.idx3-ubyte";
    private static final String TRAIN_LABELS = "./data/train-labels.idx1-ubyte";
    private static final String CUSTOM_IMAGE_DIR = "./data/"; // ME0.png ~ ME9.png 위치

    // 💾 가중치 파일 저장/로드 경로 (.model 이나 .txt 상관없이 호환됩니다)
    private static final String WEIGHT_FILE_PATH = "./weights/jeras_mnist_weights.model";

    public static void main(String[] args) {
        System.out.println("====================================================");
        System.out.println("      Jeras Integration Master Controller v2.5      ");
        System.out.println("====================================================");

        // 🌟 [모드 스위치] 원하는 작업의 플래그를 선택해서 켜세요!
        boolean runTraining = false;    // 처음부터 가상 스레드로 다시 학습시키려면 true
        boolean runInference = true;   // 저장된 모델로 내 손글씨(ME*.png)를 맞추려면 true

        // ------------------------------------------------------------------
        // 🔥 1. 학습 모드 실행
        // ------------------------------------------------------------------
        if (runTraining) {
            executeTrainingSession();
        }

        // ------------------------------------------------------------------
        // 👁️ 2. 추론 모드 실행 (유저가 레이어 몰라도 파일만 읽어 구조 자동 복원)
        // ------------------------------------------------------------------
        if (runInference) {
            executeInferenceSession();
        }
    }

    /**
     * 🏋️ [세션 A] MNIST 데이터셋 기반 모델 학습 및 가중치 파일 덤프 추출
     */
    private static void executeTrainingSession() {
        System.out.println("\n🚀 [SESSION START] 순정 가속 엔진 기반 가상 스레드 훈련을 시작합니다.");

        int inputShape = 784;
        int batchSize = 100;
        int epochs = 15;
        float learningRate = 0.0005f;
        int totalSamples = 60000;
        int numBatches = totalSamples / batchSize;

        // 1. Keras 스타일 아키텍처 가변 빌드
        Sequential model = new Sequential(inputShape);
        model.add(new Dense(512, "relu"));
        model.add(new Dense(128, "relu"));
        model.add(new Dense(10, "softmax"));

        model.compile(learningRate, "categorical_crossentropy", "accuracy");

        // 2. DataLoader 스펙에 맞게 객체 생성 및 주입
        Augmentation augmenter = new Augmentation();

        // 🛠️ 정정 완료: (batchSize, inputDim, augmenter) 시그니처 일치
        try (DataLoader dataLoader = new DataLoader(batchSize, inputShape, augmenter)) {

            System.out.println("🔄 데이터 로더 파이프라인을 가상 스레드(Virtual Thread)로 예열 중...");
            // 🛠️ 정정 완료: startAsyncLoading 메서드로 정확하게 트리거 호출
            dataLoader.startAsyncLoading(TRAIN_IMAGES, TRAIN_LABELS, totalSamples, epochs);

            // 가상 스레드 모드로 고속 동적 fit 가동
            model.fit(dataLoader, numBatches, epochs);

            // 가중치 아카이브 저장 폴더 강제 생성 가드
            java.nio.file.Files.createDirectories(java.nio.file.Paths.get("./weights"));

            // 완성 가중치 마크다운 덤프 진행
            model.saveWeights(WEIGHT_FILE_PATH);
            System.out.println("✅ [학습 성공] 모델 학습 및 가중치 저장이 완료되었습니다.");

        } catch (Exception e) {
            System.err.println("❌ [학습 에러] 훈련 도중 예외가 발생했습니다: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 🎯 [세션 B] 저장된 가중치를 파싱하여 레이어 자동 복원 후 커스텀 손글씨 예측 진행
     */
    private static void executeInferenceSession() {
        System.out.println("\n🎯 [SESSION START] 가중치 파일 자동 분석 및 실시간 이미지 추론을 시작합니다.");

        // 🌟 Keras DX의 정수: 아키텍처 선언 없이 껍데기만 만듭니다!
        Sequential model = new Sequential(784);

        try {
            // 파일 내부 명세를 읽어 레이어 개수와 결합 가중치를 실시간 동적 할당
            model.loadWeights(WEIGHT_FILE_PATH);
        } catch (IOException e) {
            System.err.println("❌ [로드 실패] 지정된 경로에서 가중치 파일을 찾을 수 없습니다: " + e.getMessage());
            return;
        }

        System.out.println("🔮 ME0.png ~ ME9.png 커스텀 이미지 전처리 및 타겟팅 로드...");
        int totalTests = 10;
        int correctTests = 0;

        for (int t = 0; t < totalTests; t++) {
            String imgName = "ME" + t + ".png";
            File imgFile = new File(CUSTOM_IMAGE_DIR + imgName);

            if (!imgFile.exists()) {
                System.out.printf("⚠️ [%s] 파일 누락으로 추론을 스킵합니다. (경로: %s)%n", imgName, imgFile.getAbsolutePath());
                continue;
            }

            try {
                // PNG 이미지를 MNIST 전용 784 평면 피처 배열로 변환
                float[] pixelFeatures = loadAndPreprocessImage(imgFile);

                // 유도리 엔진으로 전방 전개 (변수명 오타 해결)
                float[] probabilities = model.predict(pixelFeatures);

                // 최고 점수의 예측 인덱스 도출
                int predictedDigit = 0;
                float maxProb = -1f;
                for (int o = 0; o < probabilities.length; o++) {
                    if (probabilities[o] > maxProb) {
                        maxProb = probabilities[o];
                        predictedDigit = o;
                    }
                }

                boolean isCorrect = (predictedDigit == t);
                if (isCorrect) correctTests++;

                System.out.printf("📄 파일: %-8s | 실제 정답: %d | 모델 예측: %d | 신뢰도: %.2f%% | 결과: %s%n",
                        imgName, t, predictedDigit, maxProb * 100f, isCorrect ? "⭕ 정답" : "❌ 오답");

            } catch (IOException e) {
                System.err.printf("❌ [%s] 추론 처리 중 예외 발생: %s%n", imgName, e.getMessage());
            }
        }

        System.out.println("====================================================");
        System.out.printf("📊 최종 커스텀 이미지 검증 결과: %d/%d 맞춤 (정확도: %.1f%%)%n",
                correctTests, totalTests, ((float)correctTests / totalTests) * 100f);
        System.out.println("====================================================");
    }

    /**
     * 🛠IO [PNG 이미지 전처리 머신] Custom PNG -> 28x28 흑백 변환 -> MNIST 표준화(0.0~1.0)
     */
    private static float[] loadAndPreprocessImage(File file) throws IOException {
        BufferedImage originalImage = ImageIO.read(file);
        BufferedImage resizedImage = new BufferedImage(28, 28, 10);        java.awt.Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, 28, 28, null);
        g.dispose();

        float[] flattened = new float[784];

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = resizedImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gVal = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                float gray = (r + gVal + b) / 3.0f;

                // 🌟 중요: 흰 바탕에 검은 글씨 이미지라면 그대로 (255f - gray) 사용하여 반전.
                // 만약 검은 바탕에 흰 글씨로 그리셨다면 그냥 gray / 255f 로 수정하세요!
                float normalized = gray / 255f;

                flattened[y * 28 + x] = normalized;
            }
        }
        return flattened;
    }
}