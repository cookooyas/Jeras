package jeras.sandbox;

import jeras.core.Tensor;
import jeras.layers.Dense;
import jeras.models.Sequential;
import jeras.utils.DataLoader;

public class MnistQuickInference {
    public static void main(String[] args) throws Exception {
        System.out.println("📂 [Jeras v1.4] 구워진 뇌세포 복원 및 실전 추론(Inference) 시작");

        // 1. 훈련할 때와 완벽히 동일한 구조(Architecture)로 해골물(뼈대)을 만듭니다.
        // 🌟 Keras 스타일 생성자를 사용하여 Sequential이 알아서 가중치 차원을 매핑하도록 유도합니다.
        Dense layer1 = new Dense(256, "relu");
        Dense layer2 = new Dense(128, "relu");
        Dense layer3 = new Dense(10, "softmax");

        Sequential model = new Sequential(784);
        model.add(layer1);
        model.add(layer2);
        model.add(layer3);

        // 2. 10에포크 동안 피땀 흘려 저장한 가중치 파일(.txt)을 로드하여 뼈대에 살을 붙입니다.
        String modelPath = "data/jeras_mnist_model.txt";
        model.loadWeights(modelPath);

        // 3. 추론에 사용할 테스트 데이터셋 로드 (원하는 샘플 수만큼 가볍게 로딩)
        // 여기서는 검증을 위해 train 데이터에서 앞단 100장만 가져와 테스트해 봅니다.
        int testSamples = 100;
        int batchSize = 1; // 한 장씩 꼼꼼하게 추론하기 위해 배치 크기를 1로 설정

        DataLoader dataLoader = new DataLoader(batchSize, 784);
        dataLoader.startAsyncLoading("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", testSamples, 1);

        int correctPredictions = 0;

        System.out.println("\n🎯 --- 실전 개별 이미지 추론 테스트 시작 ---");

        for (int i = 1; i <= testSamples; i++) {
            DataLoader.DataBatch batch = dataLoader.nextBatch();

            // 🌟 모델의 순전파(predict)만 가동하여 0~9까지의 확률 분포를 얻습니다.
            Tensor output = model.predict(batch.x());

            // AI의 예측값(가장 확률이 높은 인덱스) 찾기
            int predictedDigit = 0;
            float maxProb = output.data[0];
            for (int o = 1; o < 10; o++) {
                if (output.data[o] > maxProb) {
                    maxProb = output.data[o];
                    predictedDigit = o;
                }
            }

            // 실제 정답(Label) 찾기 (원핫 인코딩에서 1인 위치 찾기)
            int actualDigit = 0;
            for (int o = 0; o < 10; o++) {
                if (batch.y().data[o] == 1.0f) {
                    actualDigit = o;
                    break;
                }
            }

            // 결과 비교 및 출력 (상위 10개만 콘솔에 디테일하게 출력)
            if (i <= 10) {
                System.out.printf("[%02d번 이미지] AI 예측: %d (확률: %.2f%%) | 실제 정답: %d -> %s%n",
                        i, predictedDigit, maxProb * 100f, actualDigit,
                        (predictedDigit == actualDigit) ? "정답! 🎉" : "오답.. 😭");
            }

            if (predictedDigit == actualDigit) {
                correctPredictions++;
            }
        }

        // 4. 최종 정확도(Accuracy) 산출
        float accuracy = ((float) correctPredictions / testSamples) * 100f;
        System.out.println("----------------------------------------");
        System.out.printf("📊 [최종 검증 결과] 총 %d장 중 %d장 맞춤 | 모델 정확도: %.2f%%%n",
                testSamples, correctPredictions, accuracy);

        dataLoader.close();
    }
}