package jeras.layers;

import jeras.core.Tensor;

/**
 * 🗺️ [Jeras Layer] 프레임워크 내 모든 레이어가 상속받아야 하는 표준 인터페이스 (Keras Layer Base 오마주)
 * 이 계약서를 통해 Sequential 모델은 내부 레이어의 종류에 관계없이 일관된 전파 구조를 가집니다.
 */
public interface Layer {

    /**
     * 모델에 레이어가 추가되거나 컴파일될 때, 앞단 레이어의 출력 차원을 기반으로
     * 현재 레이어의 가중치 메모리를 동적으로 할당하고 초기화합니다.
     */
    void initialize(int inputDim);

    /**
     * 현재 레이어가 출력하는 데이터의 최종 차원(노드 수)을 반환합니다.
     * 다음 레이어가 자신의 입력 차원을 자동으로 빌드업할 때 참조합니다.
     */
    int getOutputDim();

    /**
     * ⚡ [Forward Propagation]
     * 입력 텐서를 받아 레이어 고유의 수학적 연산 및 활성화 함수를 통과시킨 후 결과 텐서를 반환합니다.
     * 연산 그래프 추상을 위해 내부적으로 입력값 캐싱이 수행되어야 합니다.
     */
    Tensor forward(Tensor input);

    /**
     * ⚡ [Backpropagation]
     * 출력단에서 전달된 오차 그라디언트를 받아 내부 가중치를 조율하고,
     * 앞단 레이어로 전달할 역방향 오차 그라디언트 텐서를 연쇄적으로 계산하여 반환합니다.
     * 🌟 Keras 스타일 격리를 위해 원본 입력 Tensor 파라미터를 완벽히 제거했습니다.
     */
    Tensor backward(Tensor lossGrad, float lr);
}