package jeras.utils;

import java.util.Random;

/**
 * 🌀 [Jeras Utility] 이미지 실시간 데이터 증강(Data Augmentation) 엔진
 * Keras의 ImageDataGenerator 또는 RandomFlip/RandomRotation 레이어의 동작 철학을 오마주했습니다.
 * 2차원 공간 기하학 연산(역행렬 매핑)을 통해 매 에포크마다 뇌세포에게 매운맛 변형 문제를 출제합니다.
 */
public class Augmentation {
    private static final Random rand = new Random();

    // 🌟 인스턴스 설정을 통해 28x28 고정을 탈피하고 확장성 확보
    private final int width;
    private final int height;
    private final float maxRotationAngle; // 도(Degree) 단위 최대 회전 반경
    private final int maxShift;           // 픽셀 단위 최대 이동 반경

    /**
     * 기본 생성자: MNIST(28x28) 표준 사양 및 기본 변형 값 셋업
     */
    public Augmentation() {
        this(28, 28, 15.0f, 2);
    }

    /**
     * 커스텀 생성자: 향후 CNN 확장 및 다른 크기의 이미지 데이터셋 대응을 위한 설계
     */
    public Augmentation(int width, int height, float maxRotationAngle, int maxShift) {
        this.width = width;
        this.height = height;
        this.maxRotationAngle = maxRotationAngle;
        this.maxShift = maxShift;
    }

    /**
     * 🌟 실시간 이미지 괴롭히기 가동 (설정된 규격 기반 동적 연산)
     */
    public float[] augment(float[] original) {
        int totalPixels = width * height;
        float[] augmented = new float[totalPixels];

        // 설정된 최대 반경 내에서 무작위 각도 결정 및 라디안 변환 (예: -15도 ~ +15도)
        float angle = (float) Math.toRadians((rand.nextFloat() * (maxRotationAngle * 2.0f)) - maxRotationAngle);

        // 설정된 최대 픽셀 내에서 무작위 평행 이동량 결정 (예: -2 ~ +2)
        int shiftX = rand.nextInt((maxShift * 2) + 1) - maxShift;
        int shiftY = rand.nextInt((maxShift * 2) + 1) - maxShift;

        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);

        // 이미지의 수학적 중심점 계산 (28 기준 13.5f)
        float centerX = (width - 1.0f) / 2.0f;
        float centerY = (height - 1.0f) / 2.0f;

        // 2차원 공간 회전 및 이동 변환 (역행렬 매핑 방식으로 홀 현상/픽셀 깨짐 방지)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 중심점을 기점으로 좌표 평면 이동
                float cx = x - centerX;
                float cy = y - centerY;

                // 1. 회전 변환 적용
                float rotX = cx * cos - cy * sin;
                float rotY = cx * sin + cy * cos;

                // 2. 중심점 복원 및 평행 이동 적용
                int srcX = Math.round(rotX + centerX) - shiftX;
                int srcY = Math.round(rotY + centerY) - shiftY;

                // 3. 동적으로 계산된 격자 범위를 벗어나지 않는 경우에만 픽셀 값 복사 (외곽은 자동으로 0.0)
                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    augmented[y * width + x] = original[srcY * width + srcX];
                }
            }
        }
        return augmented;
    }

    // 🌟 [하위 호환성 유지용 정적 메서드]
    // 기존에 작성된 훈련 루프나 테스트 코드가 깨지지 않도록 static 브릿지를 남겨둡니다.
    public static float[] augment(float[] original, int width, int height) {
        return new Augmentation(width, height, 15.0f, 2).augment(original);
    }
}