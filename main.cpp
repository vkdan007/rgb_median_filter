#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <sycl/sycl.hpp>
#include "processImageData.h"
#include "medianFilter.h"
#include "medianFilterGPU.h"


bool compare_data(const uint8_t* A, const uint8_t* B, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (A[i] != B[i]) return false;
    return true;
}


void warmupGPU(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
    });
    q.wait();
}

int main() {

    BMP inputBMP;
    const int ITERATIONS = 100; 

    std::string inputPath = "img/noise/";
    std::string filename  = "noisy.bmp";
 

    if (!inputBMP.ReadFromFile((inputPath + filename).c_str())) {
        std::cerr << "Не удалось открыть файл: " << inputPath + filename << std::endl;
        return 1;
    }

    const int w = inputBMP.TellWidth();
    const int h = inputBMP.TellHeight();
    std::cout << "Изображение: " << w << " x " << h << " px" << std::endl;

    size_t sz = (size_t)w * h;

    uint8_t* inputR = new uint8_t[sz];
    uint8_t* inputG = new uint8_t[sz];
    uint8_t* inputB = new uint8_t[sz];

    load_rgb_from_bmp(inputBMP, inputR, inputG, inputB);


    uint8_t* outR_cpu = new uint8_t[sz];
    uint8_t* outG_cpu = new uint8_t[sz];
    uint8_t* outB_cpu = new uint8_t[sz];

    auto t0 = std::chrono::high_resolution_clock::now();
    MedianFilter::median_filter_3x3_rgb(
        inputR, inputG, inputB,
        outR_cpu, outG_cpu, outB_cpu,
        w, h, w);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Single thread:  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    sycl::queue q;
    std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    warmupGPU(q);

   
    uint8_t* outR_gpu1 = new uint8_t[sz];
    uint8_t* outG_gpu1 = new uint8_t[sz];
    uint8_t* outB_gpu1 = new uint8_t[sz];

   
    MedianFilterGPU::median_filter_3x3_rgb_v1(
        inputR, inputG, inputB,
        outR_gpu1, outG_gpu1, outB_gpu1,
        w, h, w, q);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        MedianFilterGPU::median_filter_3x3_rgb_v1(
            inputR, inputG, inputB,
            outR_gpu1, outG_gpu1, outB_gpu1,
            w, h, w, q);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU v1 (naive): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() / ITERATIONS
              << " ms (среднее за " << ITERATIONS << " итераций)" << std::endl;

   
    uint8_t* outR_gpu2 = new uint8_t[sz];
    uint8_t* outG_gpu2 = new uint8_t[sz];
    uint8_t* outB_gpu2 = new uint8_t[sz];

    MedianFilterGPU::median_filter_3x3_rgb_v2(
        inputR, inputG, inputB,
        outR_gpu2, outG_gpu2, outB_gpu2,
        w, h, w, q);

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        MedianFilterGPU::median_filter_3x3_rgb_v2(
            inputR, inputG, inputB,
            outR_gpu2, outG_gpu2, outB_gpu2,
            w, h, w, q);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU v2 (shared):"
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() / ITERATIONS
              << " ms (среднее за " << ITERATIONS << " итераций)" << std::endl;

   
    bool ok_v1 = compare_data(outR_cpu, outR_gpu1, sz)
              && compare_data(outG_cpu, outG_gpu1, sz)
              && compare_data(outB_cpu, outB_gpu1, sz);

    bool ok_v2 = compare_data(outR_cpu, outR_gpu2, sz)
              && compare_data(outG_cpu, outG_gpu2, sz)
              && compare_data(outB_cpu, outB_gpu2, sz);

    std::cout << "\nПроверка корректности:" << std::endl;
    std::cout << "  CPU == GPU v1: " << (ok_v1 ? "OK" : "FAILED") << std::endl;
    std::cout << "  CPU == GPU v2: " << (ok_v2 ? "OK" : "FAILED") << std::endl;

    assert(ok_v1 && "GPU v1 дал неверный результат!");
    assert(ok_v2 && "GPU v2 дал неверный результат!");


    BMP outputBMP;
    create_BMP_rgb(outputBMP, w, h, outR_gpu2, outG_gpu2, outB_gpu2);

    std::string outputPath = "img/filtered/";
    std::string outFilename = outputPath + "filtered";
    outputBMP.WriteToFile(outFilename.c_str());
    std::cout << "\nОтфильтрованное изображение сохранено: " << outFilename << std::endl;

    delete[] inputR; delete[] inputG; delete[] inputB;
    delete[] outR_cpu; delete[] outG_cpu; delete[] outB_cpu;
    delete[] outR_gpu1; delete[] outG_gpu1; delete[] outB_gpu1;
    delete[] outR_gpu2; delete[] outG_gpu2; delete[] outB_gpu2;

    return 0;
}
