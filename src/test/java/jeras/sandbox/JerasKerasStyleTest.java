//package jeras.sandbox;
//
//import jeras.core.Tensor;
//import jeras.layers.Dense;
//import jeras.models.Sequential;
//import java.util.Arrays;
//
//public class JerasKerasStyleTest {
//    public static void main(String[] args) {
//        System.out.println("==================================================");
//        System.out.println("   Jeras Framework v1.0 - Keras Style Prediction  ");
//        System.out.println("==================================================");
//
//        // 1. 가상의 MNIST 손글씨 데이터 생성 (배치 크기: 1, 이미지 크기: 28x28 = 784)
//        Tensor mockImage = new Tensor(1, 784);
//        mockImage.initRandom(); // 무작위 픽셀 데이터 채우기
//
//        // 2. Keras 감성 100% 선언적 모델 빌드 (컨베이어 벨트 조립)
//        // 사용자는 내부 행렬 크기 연산을 신경 쓸 필요가 없습니다! Jeras가 알아서 매칭합니다.
//        Sequential model = new Sequential(784); // Input Layer: 784개의 픽셀 입력
//        model.add(new Dense(128, "relu"));       // Hidden Layer 1: 128개 뉴런 + ReLU 가속 활성화
//        model.add(new Dense(64, "relu"));        // Hidden Layer 2: 64개 뉴런 + ReLU 가속 활성화
//        model.add(new Dense(10, "softmax"));     // Output Layer: 최종 숫자 0~9 확률 변환 (Softmax)
//
//        System.out.println("[INFO] 순수 자바 25 아키텍처 모델 빌드 완료.");
//        System.out.println("[INFO] Vector API 가속 기반 추론(Predict)을 시작합니다...\n");
//
//        // 3. 추론 실행 및 정밀 시간 측정
//        long startTime = System.nanoTime();
//        Tensor outputPrediction = model.predict(mockImage);
//        long endTime = System.nanoTime();
//
//        // 4. 결과 리포팅
//        double durationMs = (endTime - startTime) / 1_000_000.0;
//
//        System.out.println("================== [ 추론 결과 ] ==================");
//        System.out.printf("⚡ 순수 자바 가속 연산 소요 시간: %.4f ms%n", durationMs);
//        System.out.println("--------------------------------------------------");
//        System.out.println("💡 숫자별 분류 예측 확률 (0부터 9까지):");
//
//        // Softmax를 거쳤기 때문에 10개 칸의 총합은 정확히 1.0(100%)이 됩니다.
//        float totalProb = 0f;
//        int bestClass = 0;
//        float maxProb = -1f;
//
//        for (int i = 0; i < outputPrediction.data.length; i++) {
//            float prob = outputPrediction.data[i];
//            totalProb += prob;
//            System.out.printf("   [숫자 %d]: %.2f%%%n", i, prob * 100);
//
//            if (prob > maxProb) {
//                maxProb = prob;
//                bestClass = i;
//            }
//        }
//
//        System.out.println("--------------------------------------------------");
//        System.out.printf("🎯 Jeras 모델의 최종 판단: 이 사진은 [ %d ] 일 확률이 가장 높습니다! (확률: %.2f%%)%n",
//                bestClass, maxProb * 100);
//        System.out.printf("📊 전확률 검증 (총합이 100%%인가?): %.1f%%%n", totalProb * 100);
//        System.out.println("==================================================");
//    }
//}