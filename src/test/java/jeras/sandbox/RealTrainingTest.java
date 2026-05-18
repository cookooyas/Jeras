//package jeras.sandbox;
//
//import jeras.core.Tensor;
//import jeras.layers.Dense;
//import java.util.Arrays;
//
//public class RealTrainingTest {
//    public static void main(String[] args) {
//        System.out.println("=== Jeras 인공지능 진짜 학습 능력 검증 ===");
//
//        // 1. 단일 레이어 생성 (입력 4개 -> 출력 3개 뉴런)
//        // 목표: 무조건 0번 클래스(정답)의 확률을 100%로 만들도록 학습시키기
//        Dense layer = new Dense(3, "softmax");
//        layer.initialize(4);
//
//        // 가상의 입력 데이터 (4차원 특성)
//        Tensor input = new Tensor(1, 4);
//        Arrays.fill(input.data, 1.0f); // [1.0, 1.0, 1.0, 1.0]
//
//        // 우리가 원하는 완벽한 정답 레이블 (0번 클래스가 100%여야 함)
//        float[] target = {1.0f, 0.0f, 0.0f};
//        float learningRate = 0.1f; // 학습 속도 조절 조율자
//
//        System.out.println("\n--- 훈련 시작 전 모델의 상태 ---");
//        Tensor initOutput = layer.forward(input);
//        System.out.println("처음 예측 확률: " + Arrays.toString(initOutput.data));
//
//        System.out.println("\n[학습 중...] 오차를 계산하고 가중치를 깎아 나갑니다 (총 50회 수행)");
//
//        // 훈련 루프 실행
//        for (int epoch = 1; epoch <= 50; epoch++) {
//            // 1. 순전파 (현재 상태로 예측)
//            Tensor output = layer.forward(input);
//
//            // 2. 오차(Loss Gradient) 계산: (예측값 - 실제 정답)
//            Tensor errorGradient = new Tensor(1, 3);
//            for (int i = 0; i < 3; i++) {
//                errorGradient.data[i] = output.data[i] - target[i];
//            }
//
//            // 3. 역전파 수행 (가중치 깎기!)
//            layer.backward(input, errorGradient, learningRate);
//
//            if (epoch % 10 == 0) {
//                System.out.printf("   [반복 %2d회차] 현재 예측 확률: [%.2f%%, %.2f%%, %.2f%%]%n",
//                        epoch, output.data[0]*100, output.data[1]*100, output.data[2]*100);
//            }
//        }
//
//        System.out.println("\n--- 훈련 종료 후 모델의 상태 ---");
//        Tensor finalOutput = layer.forward(input);
//        System.out.println("최종 예측 확률: " + Arrays.toString(finalOutput.data));
//        System.out.println("================================================");
//    }
//}