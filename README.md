# Hybrid OpenMP+CUDA Image Segmentation

![Image Segmentation Banner](https://user-images.githubusercontent.com/12345678/123456789-abcdef.png)

This project provides a high-performance image segmentation solution using a hybrid approach that combines **CPU parallelism (OpenMP)** with **GPU acceleration (CUDA)**. The core of the segmentation is based on the **k-means clustering** algorithm, which efficiently groups image pixels into a specified number of clusters.

## 🚀 Features

- **Hybrid Parallelism**: Leverages both OpenMP and CUDA to maximize performance.
- **K-Means Clustering**: Implements the k-means algorithm for effective image segmentation.
- **Performance Benchmarking**: Includes scripts to measure and analyze performance.
- **Easy to Use**: Simple steps to compile and run the segmentation on your own images.

## Prerequisites

Before you begin, ensure you have the following installed:

- **NVIDIA GPU**: With CUDA toolkit installed.
- **C++ Compiler**: `g++`
- **OpenCV**: `libopencv-dev`
- **OpenMP**: `libomp-dev`

You can check for GPU availability with:
```bash
nvidia-smi
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shadow-112/Image-Segmentation-K-Means.git
    cd Image-Segmentation-K-Means
    ```
2.  **Install dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y libopencv-dev libomp-dev
    ```

## ⚙️ How to Run

1.  **Compile the program:**
    The compilation step for the CUDA and OpenMP program should be provided here. For example:
    ```bash
    nvcc -o kmeans Image-Segmentation.cu -Xcompiler -fopenmp -lopencv_core -lopencv_highgui -lopencv_imgproc
    ```
    *(Note: The user should provide the correct compilation command)*

2.  **Run the segmentation:**
    Execute the compiled program with your desired image and number of clusters.
    ```bash
    ./kmeans <input_image> <output_image> <num_clusters>
    ```
    For example:
    ```bash
    ./kmeans Lenna.png output_5.png 5
    ./kmeans Lenna.png output_16.png 16
    ```

## 📊 Results and Benchmarking

### Displaying Segmented Images

You can visualize the results using the provided Python script snippet.

```python
import cv2
import matplotlib.pyplot as plt

def display_image(title, path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display original and segmented images
display_image("Original Image", "Lenna.png")
display_image("5 Clusters", "output_5.png")
display_image("16 Clusters", "output_16.png")
```

### Performance

The repository includes code to benchmark the performance for different numbers of clusters.

```python
import time
import subprocess
import matplotlib.pyplot as plt

cluster_counts = [2, 4, 8, 16, 32]
times = []

for k in cluster_counts:
    start_time = time.time()
    command = f"./kmeans Lenna.png benchmark_{k}.png {k}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"{k} clusters: {elapsed:.2f} seconds")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(cluster_counts, times, 'o-', markersize=8)
plt.xlabel('Number of Clusters')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance vs. Cluster Count')
plt.grid(True)
plt.show()
```

## 💡 How It Works

The k-means algorithm is an iterative clustering method. For image segmentation, it works as follows:

1.  **Initialization**: `k` cluster centroids are randomly initialized.
2.  **Assignment Step**: Each pixel in the image is assigned to the nearest centroid based on color similarity (e.g., Euclidean distance in RGB space).
3.  **Update Step**: The centroids are recalculated as the mean of all pixels assigned to them.
4.  **Iteration**: Steps 2 and 3 are repeated until the cluster assignments no longer change significantly, or a maximum number of iterations is reached.

This project parallelizes the assignment step on the GPU using CUDA, as it is the most computationally intensive part of the algorithm. OpenMP is used to parallelize other tasks on the CPU.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---
