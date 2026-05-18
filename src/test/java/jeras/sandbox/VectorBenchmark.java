//package jeras.sandbox;
//
//import jeras.core.Tensor;
//
//public class VectorBenchmark {
//    public static void main(String[] args) {
//        System.out.println("=== Jeras 하드웨어 가속 성능 검증 벤치마크 ===");
//
//        // MNIST 이미지 1000개를 한 번에 처리하는 크기의 거대 행렬 정의 (1000, 784)
//        int batchSize = 1000;
//        int inputDim = 784;
//        int outputDim = 128; // 은닉층 뉴런 수
//
//        Tensor input = new Tensor(batchSize, inputDim);
//        Tensor weights = new Tensor(inputDim, outputDim);
//        Tensor bias = new Tensor(1, outputDim);
//
//        input.initRandom();
//        weights.initRandom();
//
//        System.out.println("대규모 행렬 곱 연산 시작... (차원: [" + batchSize + "x" + inputDim + "] * [" + inputDim + "x" + outputDim + "])");
//
//        // JVM 웜업(Warm-up) 및 실행 속도 측정
//        // 자바는 JIT 컴파일러가 코드를 최적화할 시간이 필요하므로 5번 반복 측정합니다.
//        for (int iter = 1; iter <= 5; iter++) {
//            long startTime = System.nanoTime();
//            Tensor result = input.matMulAndAddBias(weights, bias);
//            long endTime = System.nanoTime();
//
//            double durationMs = (endTime - startTime) / 1_000_000.0;
//            System.out.printf("[%d회차 실행] 연산 소요 시간: %.2f ms (결과 행렬 크기: %dx%d)%n",
//                    iter, durationMs, result.rows, result.cols);
//        }
//
//        System.out.println("\n순수 자바 25 가속 엔진 정상 작동 확인 완료!");
//    }
//}