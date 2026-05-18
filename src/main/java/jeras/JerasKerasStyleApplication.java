package jeras;

import jeras.core.Tensor;
import jeras.models.Sequential;
import jeras.layers.Dense;
import jeras.utils.Augmentation;
import jeras.utils.DataLoader;

public class JerasKerasStyleApplication {
    public static void main(String[] args) {
        System.out.println("====================================================");
        System.out.println("🚀 Welcome to Jeras (Java Keras Clone Framework v1.0)");
        System.out.println("====================================================\n");

        // 🎛️ 하이퍼파라미터 셋업
        int inputDim = 784;    // MNIST 이미지 28x28
        int batchSize = 100;
        int epochs = 5;
        float lr = 0.01f;

        // 📂 MNIST 데이터 경로 설정 (환경에 맞게 수정)
        String trainImages = "data/train-images.idx3-ubyte";
        String trainLabels = "data/train-labels.idx1-ubyte";
        int totalSamples = 60000;
        int numBatches = totalSamples / batchSize;

        // 1. 🌀 데이터 증강(Augmentation) 옵션 및 파이프라인 빌드 (Keras ImageDataGenerator 감성)
        Augmentation dataAugmenter = new Augmentation(28, 28, 12.0f, 2);

        // 2. 🚚 가상 스레드 기반 초고속 비동기 데이터 로더 가동 (Prefetch 팩토리)
        System.out.println("[Jeras] 백그라운드 가상 스레드 데이터 파이프라인 가동 시작...");
        DataLoader trainLoader = new DataLoader(batchSize, inputDim, dataAugmenter);
        trainLoader.startAsyncLoading(trainImages, trainLabels, totalSamples, epochs);

        // 3. 🧱 Keras 100% 싱크로율 레이어 레고 블록 쌓기
        System.out.println("[Jeras] 신경망 아키텍처 빌드 중...");
        Sequential model = new Sequential(inputDim); // 입력 차원 정의하며 시작

        model.add(new Dense(512, "relu"));    // 은닉층 1: He 초기화 + ReLU 가속
        model.add(new Dense(128, "relu"));    // 은닉층 2: He 초기화 + ReLU 가속
        model.add(new Dense(10, "softmax"));  // 출력층: Softmax 확률 분포 도출

        // 4. 🌟 [Keras 오마주] model.compile()
        // 옵티마이저 학습률전략 및 손실 수식을 사령탑에 바인딩
        System.out.println("[Jeras] 모델 컴파일 완료.");
        model.compile(lr, "categorical_crossentropy", "accuracy");

        System.out.println("\n----------------------------------------------------");
        // 5. 🏋️ [Keras 오마주] model.fit()
        // 이 한 줄로 미니배치, 비동기 프리페칭, 포워드, 백워드 자동 연쇄 체인 작동!
        model.fit(trainLoader, numBatches, epochs);
        System.out.println("----------------------------------------------------\n");

        // 6. 💾 [Keras 오마주] model.save_weights()
        // 훈련이 끝난 고귀한 가중치 텐서들을 디스크 파일로 안전하게 마크다운
        try {
            String weightPath = "models/jeras_mnist_weights.txt";
            model.saveWeights(weightPath);
        } catch (Exception e) {
            System.err.println("⚠️ 가중치 저장 중 예외 발생: " + e.getMessage());
        }

        // 7. 🧼 리소스 안전 해제 (가상 스레드 풀 종료)
        trainLoader.close();
        System.out.println("\n🎉 Jeras 우아한 트레이닝 세션이 성공적으로 종료되었습니다.");
    }
}