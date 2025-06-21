#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <omp.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error (" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

struct Pixel {
    unsigned char r, g, b;
};

__device__ float euclidean_distance(Pixel a, Pixel b) {
    return sqrtf((a.r - b.r) * (a.r - b.r) +
                 (a.g - b.g) * (a.g - b.g) +
                 (a.b - b.b) * (a.b - b.b));
}

__global__ void assign_clusters(Pixel* pixels, Pixel* cluster_centers, int* labels,
                                int num_pixels, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pixels) {
        float min_dist = FLT_MAX;
        int cluster = 0;
        for (int j = 0; j < num_clusters; ++j) {
            float dist = euclidean_distance(pixels[i], cluster_centers[j]);
            if (dist < min_dist) {
                min_dist = dist;
                cluster = j;
            }
        }
        labels[i] = cluster;
    }
}

__global__ void compute_cluster_sums(Pixel* pixels, int* labels,
                                     float* sum_r, float* sum_g, float* sum_b,
                                     int* count, int num_pixels, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pixels) {
        int cluster = labels[i];
        atomicAdd(&sum_r[cluster], (float)pixels[i].r);
        atomicAdd(&sum_g[cluster], (float)pixels[i].g);
        atomicAdd(&sum_b[cluster], (float)pixels[i].b);
        atomicAdd(&count[cluster], 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <output_path> <num_clusters>\n";
        return 1;
    }

    std::string image_path = argv[1];
    std::string output_path = argv[2];
    int num_clusters = std::atoi(argv[3]);

    if (num_clusters <= 0 || num_clusters > 256) {
        std::cerr << "Error: Number of clusters must be between 1 and 256\n";
        return 1;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image.\n";
        return 1;
    }

    int rows = image.rows;
    int cols = image.cols;
    int num_pixels = rows * cols;

    std::cout << "Loaded image: " << image_path << " (" << rows << "x" << cols << " pixels)\n";
    std::cout << "Running K-means with " << num_clusters << " clusters\n";

    // Allocate host memory
    Pixel* h_pixels = new Pixel[num_pixels];
    Pixel* h_centers = new Pixel[num_clusters];
    int* h_labels = new int[num_pixels];

    // Convert image to RGB format
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; ++i) {
        cv::Vec3b color = image.at<cv::Vec3b>(i / cols, i % cols);
        h_pixels[i] = { color[2], color[1], color[0] }; // BGR to RGB
    }

    // Initialize random cluster centers
    srand(time(nullptr));
    for (int i = 0; i < num_clusters; ++i) {
        int idx = rand() % num_pixels;
        h_centers[i] = h_pixels[idx];
        std::cout << "Initial center " << i << ": ("
                  << (int)h_centers[i].r << ", "
                  << (int)h_centers[i].g << ", "
                  << (int)h_centers[i].b << ")\n";
    }

    // Allocate device memory
    Pixel* d_pixels;
    Pixel* d_centers;
    int* d_labels;
    float* d_sum_r;
    float* d_sum_g;
    float* d_sum_b;
    int* d_count;

    CHECK_CUDA_ERROR(cudaMalloc(&d_pixels, num_pixels * sizeof(Pixel)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centers, num_clusters * sizeof(Pixel)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, num_pixels * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum_r, num_clusters * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum_g, num_clusters * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum_b, num_clusters * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_count, num_clusters * sizeof(int)));

    // Copy pixels to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_pixels, h_pixels, num_pixels * sizeof(Pixel), cudaMemcpyHostToDevice));
    std::cout << "Copied pixels to device memory\n";

    int max_iters = 10;
    int blockSize = 256;
    int numBlocks = (num_pixels + blockSize - 1) / blockSize;

    for (int iter = 0; iter < max_iters; ++iter) {
        std::cout << "\nIteration " << iter + 1 << "/" << max_iters << "\n";

        // Copy current centers to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_centers, h_centers, num_clusters * sizeof(Pixel), cudaMemcpyHostToDevice));

        // Assign pixels to clusters
        assign_clusters<<<numBlocks, blockSize>>>(d_pixels, d_centers, d_labels, num_pixels, num_clusters);
        CHECK_CUDA_ERROR(cudaPeekAtLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        std::cout << "Cluster assignment completed\n";

        // Reset sums and counts
        CHECK_CUDA_ERROR(cudaMemset(d_sum_r, 0, num_clusters * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_sum_g, 0, num_clusters * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_sum_b, 0, num_clusters * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_count, 0, num_clusters * sizeof(int)));

        // Compute cluster sums
        compute_cluster_sums<<<numBlocks, blockSize>>>(d_pixels, d_labels,
                                                     d_sum_r, d_sum_g, d_sum_b, d_count,
                                                     num_pixels, num_clusters);
        CHECK_CUDA_ERROR(cudaPeekAtLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        std::cout << "Cluster sums computed\n";

        // Allocate host memory for results
        float* h_sum_r = new float[num_clusters]();
        float* h_sum_g = new float[num_clusters]();
        float* h_sum_b = new float[num_clusters]();
        int* h_count = new int[num_clusters]();

        // Copy results back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_sum_r, d_sum_r, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_sum_g, d_sum_g, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_sum_b, d_sum_b, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_count, d_count, num_clusters * sizeof(int), cudaMemcpyDeviceToHost));

        // Update cluster centers
        for (int j = 0; j < num_clusters; ++j) {
            if (h_count[j] > 0) {
                h_centers[j].r = static_cast<unsigned char>(h_sum_r[j] / h_count[j]);
                h_centers[j].g = static_cast<unsigned char>(h_sum_g[j] / h_count[j]);
                h_centers[j].b = static_cast<unsigned char>(h_sum_b[j] / h_count[j]);
            }
            std::cout << "Cluster " << j << ": count=" << h_count[j]
                      << ", center=(" << (int)h_centers[j].r << ", "
                      << (int)h_centers[j].g << ", " << (int)h_centers[j].b << ")\n";
        }

        // Free temporary host memory
        delete[] h_sum_r;
        delete[] h_sum_g;
        delete[] h_sum_b;
        delete[] h_count;
    }

    // Copy final labels back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, num_pixels * sizeof(int), cudaMemcpyDeviceToHost));

    // Create segmented image
    cv::Mat segmented(rows, cols, CV_8UC3);
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; ++i) {
        int cluster = h_labels[i];
        Pixel p = h_centers[cluster];
        segmented.at<cv::Vec3b>(i / cols, i % cols) = cv::Vec3b(p.b, p.g, p.r); // RGB to BGR
    }

    // Save result
    if (!cv::imwrite(output_path, segmented)) {
        std::cerr << "Error: Could not save output image\n";
    } else {
        std::cout << "\nSegmentation complete. Saved result to: " << output_path << std::endl;
    }

    // Clean up
    delete[] h_pixels;
    delete[] h_centers;
    delete[] h_labels;
    cudaFree(d_pixels);
    cudaFree(d_centers);
    cudaFree(d_labels);
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_count);

    return 0;
}
