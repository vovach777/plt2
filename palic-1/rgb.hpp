#pragma once
#include <cstdint>
#include "../avx2.h"

namespace conv {

// --- Константы для convert_to_plannar_line_8to16_48 ---
alignas(32) static const uint8_t LCPI0_0[] = {
    255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255,
    0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0
};
alignas(16) static const uint8_t LCPI0_1[] = {
    128, 128, 128, 128, 128, 128, 2, 5, 8, 11, 14, 128, 128, 128, 128, 128
};
alignas(16) static const uint8_t LCPI0_2[] = {
    0, 3, 6, 9, 12, 15, 128, 128, 128, 128, 128, 1, 4, 7, 10, 13
};
alignas(32) static const uint8_t LCPI0_3[] = {
    0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255
};
alignas(16) static const uint8_t LCPI0_4[] = {
    128, 128, 128, 128, 128, 0, 3, 6, 9, 12, 15, 128, 128, 128, 128, 128
};
alignas(16) static const uint8_t LCPI0_5[] = {
    1, 4, 7, 10, 13, 128, 128, 128, 128, 128, 128, 2, 5, 8, 11, 14
};
alignas(32) static const uint8_t LCPI0_6[] = {
    0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0
};
alignas(16) static const uint8_t LCPI0_7[] = {
    128, 128, 128, 128, 128, 1, 4, 7, 10, 13, 128, 128, 128, 128, 128, 128
};
alignas(16) static const uint8_t LCPI0_8[] = {
    2, 5, 8, 11, 14, 128, 128, 128, 128, 128, 0, 3, 6, 9, 12, 15
};

// --- Константы для convert_from_plannar_line_16to8_48 ---
alignas(16) static const uint8_t LCPI1_1[] = {
    128, 0, 6, 128, 0, 7, 128, 0, 8, 128, 0, 9, 128, 0, 10, 128
};
alignas(16) static const uint8_t LCPI1_2[] = {
    5, 0, 128, 6, 0, 128, 7, 0, 128, 8, 0, 128, 9, 0, 128, 10
};
alignas(16) static const uint8_t LCPI1_3[] = {
    0, 1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10
};
alignas(32) static const uint8_t LCPI1_4[] = {
    0, 0, 0, 1, 0, 1, 0, 0, 2, 3, 2, 3, 0, 0, 4, 5,
    4, 5, 0, 0, 6, 7, 6, 7, 0, 0, 8, 9, 8, 9, 0, 0
};
alignas(32) static const uint8_t LCPI1_5[] = {
    255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255,
    255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255
};
alignas(16) static const uint8_t LCPI1_6[] = {
    128, 6, 7, 128, 8, 9, 128, 10, 11, 128, 12, 13, 128, 14, 15, 128
};
alignas(16) static const uint8_t LCPI1_7[] = {
    10, 128, 128, 11, 128, 128, 12, 128, 128, 13, 128, 128, 14, 128, 128, 15
};

// Функция 1: Прямое преобразование (упакованные 8-bit в планарные 16-bit)
void convert_to_plannar_line_8to16_48(const unsigned char* src, short* dst1, short* dst2, short* dst3) {
    simde__m256i ymm0 = simde_mm256_loadu_si256((const simde__m256i*)src);
    simde__m256i ymm1 = simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i*)(src + 32)));

    simde__m256i ymm2 = simde_mm256_load_si256((const simde__m256i*)LCPI0_0);
    ymm2 = simde_mm256_blendv_epi8(ymm1, ymm0, ymm2);

    simde__m128i xmm3 = simde_mm256_extracti128_si256(ymm2, 1);
    xmm3 = simde_mm_shuffle_epi8(xmm3, simde_mm_load_si128((const simde__m128i*)LCPI0_1));

    simde__m128i xmm2 = simde_mm256_castsi256_si128(ymm2);
    xmm2 = simde_mm_shuffle_epi8(xmm2, simde_mm_load_si128((const simde__m128i*)LCPI0_2));
    xmm2 = simde_mm_or_si128(xmm2, xmm3);

    simde__m256i ymm3 = simde_mm256_load_si256((const simde__m256i*)LCPI0_3);
    ymm3 = simde_mm256_blendv_epi8(ymm1, ymm0, ymm3);
    ymm2 = simde_mm256_cvtepu8_epi16(xmm2);

    simde__m128i xmm4 = simde_mm256_extracti128_si256(ymm3, 1);
    xmm4 = simde_mm_shuffle_epi8(xmm4, simde_mm_load_si128((const simde__m128i*)LCPI0_4));

    xmm3 = simde_mm256_castsi256_si128(ymm3);
    xmm3 = simde_mm_shuffle_epi8(xmm3, simde_mm_load_si128((const simde__m128i*)LCPI0_5));
    xmm3 = simde_mm_or_si128(xmm3, xmm4);
    ymm3 = simde_mm256_cvtepu8_epi16(xmm3);

    simde__m256i ymm4 = simde_mm256_load_si256((const simde__m256i*)LCPI0_6);
    ymm0 = simde_mm256_blendv_epi8(ymm1, ymm0, ymm4);

    simde__m128i xmm1 = simde_mm256_extracti128_si256(ymm0, 1);
    xmm1 = simde_mm_shuffle_epi8(xmm1, simde_mm_load_si128((const simde__m128i*)LCPI0_7));

    simde__m128i xmm0 = simde_mm256_castsi256_si128(ymm0);
    xmm0 = simde_mm_shuffle_epi8(xmm0, simde_mm_load_si128((const simde__m128i*)LCPI0_8));
    xmm0 = simde_mm_or_si128(xmm0, xmm1);
    ymm0 = simde_mm256_cvtepu8_epi16(xmm0);

    ymm1 = simde_mm256_sub_epi16(ymm2, ymm3);
    ymm4 = simde_mm256_sub_epi16(ymm0, ymm3);

    simde__m256i ymm5 = simde_mm256_add_epi16(ymm3, ymm3);
    ymm2 = simde_mm256_sub_epi16(ymm2, ymm5);
    ymm0 = simde_mm256_add_epi16(ymm2, ymm0);
    ymm0 = simde_mm256_srai_epi16(ymm0, 2);
    ymm0 = simde_mm256_add_epi16(ymm0, ymm3);

    simde_mm256_storeu_si256((simde__m256i*)dst1, ymm1);
    simde_mm256_storeu_si256((simde__m256i*)dst2, ymm4);
    simde_mm256_storeu_si256((simde__m256i*)dst3, ymm0);
}

// Функция 2: Обратное преобразование (планарные 16-bit обратно в упакованные 8-bit)
void convert_from_plannar_line_16to8_48(unsigned char* dst, const short* src1, const short* src2, const short* src3) {
    simde__m256i ymm0 = simde_mm256_loadu_si256((const simde__m256i*)src1);
    simde__m256i ymm1 = simde_mm256_loadu_si256((const simde__m256i*)src3); // Передается через rcx
    simde__m256i ymm2 = simde_mm256_loadu_si256((const simde__m256i*)src2); // Передается через rdx

    simde__m256i ymm3 = simde_mm256_add_epi16(ymm0, ymm2);
    ymm3 = simde_mm256_srai_epi16(ymm3, 2);

    simde__m256i ymm4 = simde_mm256_set1_epi16(255); // .LCPI1_8 (.short 255)

    simde__m256i ymm5 = simde_mm256_and_si256(ymm3, ymm4);
    simde__m128i xmm6 = simde_mm256_extracti128_si256(ymm5, 1);
    simde__m128i xmm5 = simde_mm_packus_epi16(simde_mm256_castsi256_si128(ymm5), xmm6);

    simde__m256i ymm6 = simde_mm256_and_si256(ymm1, ymm4);
    simde__m128i xmm7 = simde_mm256_extracti128_si256(ymm6, 1);
    xmm6 = simde_mm_packus_epi16(simde_mm256_castsi256_si128(ymm6), xmm7);

    ymm0 = simde_mm256_add_epi16(ymm0, ymm1);
    ymm0 = simde_mm256_sub_epi16(ymm0, ymm3);
    ymm0 = simde_mm256_and_si256(ymm0, ymm4);

    xmm7 = simde_mm256_extracti128_si256(ymm0, 1);
    simde__m128i xmm0 = simde_mm_packus_epi16(simde_mm256_castsi256_si128(ymm0), xmm7);

    xmm7 = simde_mm_shuffle_epi8(xmm0, simde_mm_load_si128((const simde__m128i*)LCPI1_1));
    xmm5 = simde_mm_sub_epi8(xmm6, xmm5);
    xmm6 = simde_mm_shuffle_epi8(xmm5, simde_mm_load_si128((const simde__m128i*)LCPI1_2));
    xmm6 = simde_mm_or_si128(xmm6, xmm7);

    xmm7 = simde_mm_unpacklo_epi8(xmm0, xmm5);
    xmm7 = simde_mm_shuffle_epi8(xmm7, simde_mm_load_si128((const simde__m128i*)LCPI1_3));

    ymm6 = simde_mm256_inserti128_si256(simde_mm256_castsi128_si256(xmm6), xmm7, 1);

    ymm1 = simde_mm256_add_epi16(ymm1, ymm2);
    ymm1 = simde_mm256_sub_epi16(ymm1, ymm3);
    ymm1 = simde_mm256_and_si256(ymm1, ymm4);

    simde__m128i xmm2 = simde_mm256_extracti128_si256(ymm1, 1);
    simde__m128i xmm1 = simde_mm_packus_epi16(simde_mm256_castsi256_si128(ymm1), xmm2);

    ymm2 = simde_mm256_permute4x64_epi64(ymm1, 68);
    ymm2 = simde_mm256_shuffle_epi8(ymm2, simde_mm256_load_si256((const simde__m256i*)LCPI1_4));

    ymm3 = simde_mm256_load_si256((const simde__m256i*)LCPI1_5);
    ymm2 = simde_mm256_blendv_epi8(ymm2, ymm6, ymm3);

    xmm0 = simde_mm_unpackhi_epi8(xmm0, xmm5);
    xmm0 = simde_mm_shuffle_epi8(xmm0, simde_mm_load_si128((const simde__m128i*)LCPI1_6));
    xmm1 = simde_mm_shuffle_epi8(xmm1, simde_mm_load_si128((const simde__m128i*)LCPI1_7));
    xmm0 = simde_mm_or_si128(xmm0, xmm1);

    simde_mm_storeu_si128((simde__m128i*)(dst + 32), xmm0);
    simde_mm256_storeu_si256((simde__m256i*)dst, ymm2);
}

}
