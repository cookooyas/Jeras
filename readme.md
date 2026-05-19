# 🚀 Jeras: Pure Java Dynamic MLP Engine

**Jeras**는 외부 의존성(ND4J, BLAS 등)을 일절 배제하고 오직 **순수 자바(JDK 25)**의 최신 스펙만으로 구현한 경량 다층 퍼셉트론(MLP) 딥러닝 엔진입니다.

파이썬 생태계의 프레임워크를 쓰지 않더라도, 최신 자바의 강력한 기능들을 유기적으로 결합하면 자바 자체만으로도 충분히 고성능의 머신러닝 라이브러리를 구축할 수 있음을 증명하기 위해 개발되었습니다.

---

## 🛠️ Core Stack & Philosophy (JDK 25 활용기)

* **⚡ 가상 스레드 기반의 비동기 파이프라인 (`Virtual Threads`)**
    - 디스크에서 MNIST 바이너리 데이터를 읽고, 이미지 증강(Augmentation)과 정규화를 집도하는 프로세스를 가상 스레드 기반의 `DataLoader`로 완전히 분리했습니다.
    - 이를 통해 훈련 루프와 데이터 공급 루프가 완벽히 비동기로 맞물려 돌며 디스크 I/O 병목을 제로로 만듭니다.
* **🧬 벡터 API 맛보기 (`jdk.incubator.vector`)**
    - JVM 내부의 자원을 효율적으로 쓰기 위해 인큐베이터 모듈을 연동하여, 대량의 행렬 연산이 동적 다층 구조 위에서도 유연하고 빠르게 가속될 수 있는 발판을 마련했습니다.
* **Keras 스타일의 유연한 DX (Developer Experience)**
    - 과거의 고정형 아키텍처 구조를 완전히 탈피하여, 사용자가 원하는 대로 은닉층을 겹겹이 쌓을 수 있는 가변형 레이어 프레임워크를 자체 구축했습니다.

---

## 📦 Quick Start (사용법)

신경망의 구조를 하드코딩할 필요 없이, 아래와 같이 직관적으로 빌드하고 컴파일하여 바로 훈련에 진입할 수 있습니다.

```java
// 1. 초기 입력 피처 개수를 지정하여 모델 생성 (MNIST: 784)
Sequential model = new Sequential(784);

// 2. 가변 Dense 레이어 유연하게 추가 (Leaky ReLU 적용)
model.add(new Dense(512, "relu"));
model.add(new Dense(128, "relu"));
model.add(new Dense(10, "softmax")); // 출력층 (0~9 분류)

// 3. 학습률 및 크로스 엔트로피 손실함수 설정
model.compile(0.0005f, "categorical_crossentropy", "accuracy");

// 4. 가상 스레드 파이프라인 로더를 점화하여 에포크 기반 fit 가동
model.fit(dataLoader, numBatches, epochs);

// 5. 완성된 명품 가중치는 텍스트 구조 파일로 영구 덤프 추출
model.saveWeights("./weights/jeras_mnist_weights.model");
```

---

## 👁️ Architecture Auto-Restoration (동적 복원 기능)

Jeras는 아키텍처 선언을 미리 해두지 않아도, 저장된 가중치 파일(.model)의 마크다운 명세를 실시간으로 파싱하여 레이어 개수와 결합 가중치 스케일을 알아서 동적 할당하고 복원합니다.
```java
// 단 한 줄의 model.add() 없이 빈 껍데기만 생성
Sequential model = new Sequential(784);

// 파일 내부의 명세를 읽어 레이어 3개와 연결 노드 구조를 자동으로 복원
model.loadWeights("./weights/jeras_mnist_weights.model");

// 복원된 유도리 엔진으로 실시간 손글씨(PNG) 추론 가동!
float[] probabilities = model.predict(pixelFeatures);
```

---

## 📊 Project Output

매 에포크마다 비동기로 찌그러트리고 노이즈를 섞은 고난도 데이터 증강(Augmentation) 환경을 견뎌내며 일반화 성능을 획득했습니다. 
그 결과, 사용자가 직접 마우스로 그린 커스텀 이미지(ME0.png ~ ME9.png) 테스트셋에서 10/10 올킬 정답(정확도 100%)을 달성하며 엔진의 정교함을 입증했습니다.