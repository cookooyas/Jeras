package jeras.utils;

import java.util.Random;

public class Augmentation {
    private static final Random rand = new Random();

    /**
     * 784차원 MNIST 이미지 배열을 무작위로 회전 및 이동시켜 변환합니다.
     */
    public static float[] augment(float[] original) {
        // 원본 훼손 방지를 위한 복사
        float[] augmented = new float[784];

        // -15도 ~ +15도 사이 무작위 회전 각도 결정 (라디안 변환)
        float angle = (float) Math.toRadians((rand.nextFloat() * 30.0) - 15.0);

        // -2픽셀 ~ +2픽셀 사이 무작위 이동량 결정
        int shiftX = rand.nextInt(5) - 2;
        int shiftY = rand.nextInt(5) - 2;

        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);

        // 2차원 공간 회전 및 이동 변환 (역행렬 매핑 방식으로 픽셀 깨짐 방지)
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                // 중심점(13.5, 13.5) 기점으로 좌표 평면 이동
                float cx = x - 13.5f;
                float cy = y - 13.5f;

                // 1. 회전 변환 적용
                float rotX = cx * cos - cy * sin;
                float rotY = cx * sin + cy * cos;

                // 2. 중심점 복원 및 평행 이동 적용
                int srcX = Math.round(rotX + 13.5f) - shiftX;
                int srcY = Math.round(rotY + 13.5f) - shiftY;

                // 3. 28x28 격자 범위를 벗어나지 않는 경우에만 픽셀 값 복사 (외곽은 자동으로 0.0)
                if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
                    augmented[y * 28 + x] = original[srcY * 28 + srcX];
                }
            }
        }
        return augmented;
    }
}