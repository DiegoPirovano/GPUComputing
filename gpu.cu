%%cuda_group_save --name "dice.cu" --group "dice"

/*
 * CUDA Average Dice Coefficient, Mean Area/Entropy, and StdDev
 *
 * This project performs a comparative analysis of medical image segmentations
 * from an HDF5 file. It implements a optimized, batch-processing
 * CUDA pipeline and compares its performance against a sequential,
 * C implementation.
 *
 *s_float_data METRICS CALCULATED (per-image, per-group):
 * 1. Mean Agreement Entropy (based on a spatial probability map).
 * 2. StdDev of L2-Normalized Segmented Area.
 * 3. Average Dice Coefficient (for Expert-Expert and Non-Expert-Non-Expert pairs).
 *
 * IMPLEMENTATION:
 * - Host data grouping (maps, vectors) uses C++ STL for simplicity.
 * - A final verification step checks GPU results against CPU results.
 */

#include "/content/GPUcomputing/utils/common.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include "hdf5.h"

#include <string>
#include <vector>
#include <map>
#include <set>


/**
 * @brief Macro for HDF5 error checking.
 */
#define checkH5Errors(val, msg) check_hdf5((val), (msg), __FILE__, __LINE__)

/**
 * @brief Helper function for checkH5Errors.
 */
void check_hdf5(herr_t result, const char* const msg, const char* const file, int const line) {
    if (result < 0) {
        fprintf(stderr, "HDF5 Error at %s:%d: %s\n", file, line, msg);
        exit(EXIT_FAILURE);
    }
}


// Epsilon prevents log(0) in entropy and division by zero
const float epsilon = 1e-6f;

/**
 * @brief Calculates the standard deviation of an array.
 * @param values      Input array.
 * @param count       The number of elements in values.
 * @return out_stdDev  returns the calculated stddev in a double.
 */
double calculateMeanStdDev(const double* values, size_t count) {
    // Check for empty lists to prevent division by zero.
    if (count == 0) {
        return 0.0;
    }
    double mean = 0.0;
    // 1: Compute the mean
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    mean = sum / count;

    // 2: Compute the sum of squared differences
    double sq_diff_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - (mean);
        sq_diff_sum += diff * diff;
    }

    // Final calculation for StdDev
    if (count > 0) {
        return sqrt(sq_diff_sum / count);
    } else {
        return 0.0;
    }
}


/**
 * @brief SEQUENTIAL Dice calculation for a single pair.
 * @param h_maskA     Pointer to mask A.
 * @param h_maskB     Pointer to mask B.
 * @param maskSize    Total number of pixels in each mask.
 * @param epsilon     A small value to prevent division by zero.
 * @return The Dice coefficient as a double.
 */
double sequentialDicePair(const unsigned char* h_maskA,
                            const unsigned char* h_maskB,
                            int maskSize,
                            float epsilon)
{
    int h_totalIntersection = 0;
    int h_totalSum = 0;

    for (int i = 0; i < maskSize; i++) {
        unsigned char valA = h_maskA[i] > 0 ? 1 : 0;
        unsigned char valB = h_maskB[i] > 0 ? 1 : 0;

        h_totalIntersection += (valA & valB); // Intersection (1 if both are 1)
        h_totalSum += valA + valB;            // Sum of pixels
    }

    return (2.0 * h_totalIntersection) / (h_totalSum + epsilon);
}

/**
 * @brief SEQUENTIAL implementation for all neded metrics.
 *
 * 1. StdDev of L2-Normalized Area:
 * - Computes the area (pixel sum) for each mask.
 * - Computes the L2 norm of this "area vector".
 * - Normalizes the area vector by this norm.
 * - Computes the StdDev of the *normalized* areas.
 *
 * 2. Mean Agreement Entropy:
 * - Creates a "probability map" by averaging all masks in the group.
 * - Computes the entropy for each pixel in this map.
 * - Returns the mean of these entropy values.
 *
 * 3. Average Dice Coefficient:
 * - Computes the Dice score for all unique pairs *within* this group.
 * - Returns the average of these Dice scores.
 *
 * @param h_group_masks      array of pointers, where each pointer points to a mask.
 * @param numMasksInGroup    The number of masks in this group.
 * @param maskSize           Total number of pixels in each mask.
 * @param out_normalized_area_stddev  Output pointer for the StdDev.
 * @param out_mean_agreement_entropy  Output pointer for the Mean Entropy.
 * @param out_average_dice            Output pointer for the Average Dice.
 */
void sequentialMaskMetrics(unsigned char** h_group_masks,
                             int numMasksInGroup,
                             int maskSize,
                             double* out_normalized_area_stddev,
                             double* out_mean_agreement_entropy,
                             double* out_average_dice)
{
    // Safety check for empty groups
    if (numMasksInGroup == 0) {
        *out_normalized_area_stddev = 0.0;
        *out_mean_agreement_entropy = 0.0;
        *out_average_dice = 0.0;
        return;
    }

    // 1. Compute Standard Deviation of Normalized Areas ---

    // Allocate memory for the list of raw areas
    double* areas = (double*)malloc(numMasksInGroup * sizeof(double));
    if (areas == NULL) { /* Handle malloc failure */ return; }

    double area_l2_norm_sq = 0.0;

    // First pass: get all raw areas
    for (int i = 0; i < numMasksInGroup; i++) {
        unsigned int area = 0;
        for (int p = 0; p < maskSize; p++) {
            if (h_group_masks[i][p] > 0) {
                area++;
            }
        }
        areas[i] = (double)area;
        area_l2_norm_sq += areas[i] * areas[i]; // Accumulate for L2 norm
    }

    double area_l2_norm = sqrt(area_l2_norm_sq) + epsilon; // Add epsilon

    // Allocate memory for the normalized areas
    double* normalized_areas = (double*)malloc(numMasksInGroup * sizeof(double));
    if (normalized_areas == NULL) { free(areas); return; }

    // Second pass: normalize the areas
    for (int i = 0; i < numMasksInGroup; i++) {
        normalized_areas[i] = areas[i] / area_l2_norm;
    }

    // Calculate StdDev of the normalized areas
    double stddev_norm_area = 0.0;
    stddev_norm_area = calculateMeanStdDev(normalized_areas, numMasksInGroup);
    *out_normalized_area_stddev = stddev_norm_area;


    // 2. Compute Mean Agreement Entropy ---

    // Allocate and zero-initialize the probability map
    double* prob_map = (double*)calloc(maskSize, sizeof(double));
    if (prob_map == NULL) { free(areas); free(normalized_areas); return; }

    // Create the probability map by summing all masks pixel-wise
    for (int p = 0; p < maskSize; p++) {
        double pixel_sum = 0.0;
        for (int i = 0; i < numMasksInGroup; i++) {
            if (h_group_masks[i][p] > 0) {
                pixel_sum++;
            }
        }
        // Average the sum to get the probability
        prob_map[p] = pixel_sum / (double)numMasksInGroup;
    }

    double total_entropy = 0.0;
    const double clip_min = 1e-8;
    const double clip_max = 1.0 - 1e-8;

    // Calculate entropy for each mask
    for (int p = 0; p < maskSize; p++) {
        double prob = prob_map[p];
        // Clip values to avoid log(0)
        if (prob < clip_min) prob = clip_min;
        if (prob > clip_max) prob = clip_max;

        // Entropy: -p*log(p) - (1-p)*log(1-p)
        total_entropy += -(1.0 - prob) * log((1.0 - prob) + clip_min) - prob * log(prob + clip_min);
    }

    *out_mean_agreement_entropy = total_entropy / (double)maskSize;


    // 3. Compute Average Dice Coefficient ---
    double total_dice_sum = 0.0;
    int num_pairs = 0;

    for (int i = 0; i < numMasksInGroup; i++) {
        for (int j = i + 1; j < numMasksInGroup; j++) {
            double dice = sequentialDicePair(h_group_masks[i],
                                               h_group_masks[j],
                                               maskSize,
                                               epsilon);
            total_dice_sum += dice;
            num_pairs++;
        }
    }

    if (num_pairs > 0) {
        *out_average_dice = total_dice_sum / num_pairs;
    } else {
        *out_average_dice = 0.0;
    }

    // Cleanup
    free(areas);
    free(normalized_areas);
    free(prob_map);
}


/**
 * @brief Holds annotation data *after* being grouped by image.
 * This is the 'value' in the groupedAnnotations map.
 */
struct Annotation {
    int h_mask_index; // Index in the global h_masks array
    int is_expert;    // 1 if expert, 0 if not
};

/**
 * @brief Holds mask info *before* grouping, used for per-mask analysis.
 * This list parallels the h_masks array.
 */
struct MaskInfo {
    std::string imageName; // The image this mask belongs to
    int is_expert;         // 1 if expert, 0 if not
};

/**
 * @brief Holds pair info for the batch kernel, used for post-processing.
 * This list parallels the Dice kernel output arrays.
 */
struct PairInfo {
    std::string imageName;      // The image this pair belongs to
    int is_expert_pair; // 0 = Non-Expert/Non-Expert, 1 = Expert/Expert
};

// --- HDF5 Helper Function

/**
 * @brief Reads a string attribute from an HDF5 group.
 *
 * This function safely reads an attribute from an HDF5 group and
 * returns it as a std::string.
 *
 * It correctly handles both fixed-length and variable-length strings,
 * which is a common point of failure. The fix for variable-length
 * strings (using attr_type as mem_type) is critical.
 *
 * @param group_id   The HDF5 identifier for the open group.
 * @param attr_name  The name of the attribute to read (e.g., "Origin").
 * @return The attribute's value as a std::string.
 */
std::string readH5StringAttribute(hid_t group_id, const char* attr_name) {
    hid_t attr_id = H5Aopen(group_id, attr_name, H5P_DEFAULT);
    checkH5Errors(attr_id, "H5Aopen");

    hid_t attr_type = H5Aget_type(attr_id);
    checkH5Errors(attr_type, "H5Aget_type");

    std::string result = "";

    if (H5Tis_variable_str(attr_type) > 0) {
        // --- Variable-length string ---
        char* var_str = NULL;

        herr_t status = H5Aread(attr_id, attr_type, &var_str);
        checkH5Errors(status, "H5Aread (var-len string)");

        if (var_str != NULL) {
            result = std::string(var_str);
            H5free_memory(var_str); //free memory
        }

    } else {
        // --- Fixed-length string ---
        hsize_t attr_size = H5Aget_storage_size(attr_id);
        if (attr_size > 0) {
            // Allocate +1 for the null terminator
            char* fixed_str = (char*)malloc(attr_size + 1);

            herr_t status = H5Aread(attr_id, attr_type, fixed_str);
            checkH5Errors(status, "H5Aread (fixed-len string)");

            fixed_str[attr_size] = '\0'; // Manually null-terminate
            result = std::string(fixed_str);
            free(fixed_str);
        }
    }

    H5Tclose(attr_type);
    checkH5Errors(H5Aclose(attr_id), "H5Aclose");

    return result;
}

// --- KERNEL PROTOTYPES ---

__global__ void calculateAreaKernel(const unsigned char* d_all_masks,
                                    int maskSize,
                                    unsigned int* d_out_areas,
                                    int numMasks);

__global__ void diceBatchKernel(const unsigned char* d_all_masks,
                                int maskSize,
                                const int* d_pair_mask_i,
                                const int* d_pair_mask_j,
                                unsigned int* d_out_intersections,
                                unsigned int* d_out_sums,
                                int numPairs);

__global__ void sumMasksKernel(const unsigned char* d_all_masks,
                               int maskSize,
                               const int* d_group_mask_indices,
                               int numMasksInGroup,
                               unsigned int* d_out_sum_map);

__global__ void entropyMapReduceKernel(const unsigned int* d_sum_map,
                                       int maskSize,
                                       int numMasksInGroup,
                                       float* d_block_entropies);

__global__ void sumReduceFloatKernel(const float* d_input, int N, float* d_output);


// --- MAIN FUNCTION ---

int main(void) {
    // HDF5 File and Group Setup ---
    const char* filename = "annotations.h5";
    const char* mask_dataset_name = "Mask_UL";
    const char* expertise_attribute_name = "expertise";
    const char* origin_attribute_name = "Origin";

    // HDF5 Loading: Pass 1 (Get mask size and group count) ---
    // We must do a two-pass load.
    // Pass 1: Find the *first* valid mask to determine its size.
    // We need this size to allocate one giant 'h_masks' buffer.

    printf("Reading masks from HDF5 file: %s\n", filename);

    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    checkH5Errors(file_id, "H5Fopen");

    // Get the total number of groups (annotations) in the file
    hsize_t numTotalGroups = 0;
    H5Gget_num_objs(file_id, &numTotalGroups);
    printf("Found %dtotal groups in file.\n", (unsigned int)numTotalGroups);

    long int maskSize = 0;
    int numValidMasks = 0; // This will count only the masks we actually load

    printf("Pass 1: Determining mask size...\n");
    for (int i = 0; i < numTotalGroups; i++) {
        // Get group name by its index
        char group_name[256];
        ssize_t name_len = H5Lget_name_by_idx(file_id, "/", H5_INDEX_NAME, H5_ITER_INC, (hsize_t)i, group_name, 256, (hid_t)0);
        if (name_len < 0) continue;

        hid_t group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
        if (group_id < 0) continue;

        // Check if the mask dataset exists in this group
        if (H5Lexists(group_id, mask_dataset_name, H5P_DEFAULT) > 0) {
            hid_t dataset_id = H5Dopen(group_id, mask_dataset_name, H5P_DEFAULT);
            if (dataset_id < 0) { H5Gclose(group_id); continue; }

            // Get the dataspace (dimensions) of the mask
            hid_t space_id = H5Dget_space(dataset_id);
            int ndims = H5Sget_simple_extent_ndims(space_id);
            if (ndims < 1) {
                H5Sclose(space_id); H5Dclose(dataset_id); H5Gclose(group_id); continue;
            }

            hsize_t dims[ndims];
            H5Sget_simple_extent_dims(space_id, dims, NULL);

            // Calculate total number of elements (pixels)
            int currentMaskSize = 1;
            for (int d = 0; d < ndims; d++) { currentMaskSize *= dims[d]; }

            H5Sclose(space_id); H5Dclose(dataset_id);

            // If we found a valid mask, store its size and break the loop
            if (currentMaskSize > 0) {
                maskSize = currentMaskSize;
                printf("Determined mask size: %ld elements\n", maskSize);
                H5Gclose(group_id);
                break;
            }
        }
        H5Gclose(group_id);
    }

    if (maskSize == 0) {
        fprintf(stderr, "Error: Could not find any valid masks with name '%s' in file.\n", mask_dataset_name);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    // --- HDF5 Loading: Pass 2 (Load all masks and attributes) ---
    // Now we allocate one large buffer on the host (CPU) to hold ALL
    // mask data contiguously.

    size_t totalMasksMemory = (size_t)numTotalGroups * maskSize * sizeof(unsigned char);
    unsigned char* h_masks = (unsigned char*)malloc(totalMasksMemory);
    if (h_masks == NULL) {
        fprintf(stderr, "Error: Failed to allocate %zu bytes for host masks.\n", totalMasksMemory);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }


    std::map<std::string, std::vector<Annotation>> groupedAnnotations;
    std::vector<MaskInfo> maskInfoList;
    std::set<std::string> uniqueImageNames;

    printf("Pass 2: Loading and grouping all valid masks and attributes...\n");

    for (int i = 0; i < numTotalGroups; i++) {
        char group_name[256];
        H5Lget_name_by_idx(file_id, "/", H5_INDEX_NAME, H5_ITER_INC, (hsize_t)i, group_name, 256, (hid_t)0);
        hid_t group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
        if (group_id < 0) continue;

        // Skip group if it doesn't have the mask we're looking for
        if (H5Lexists(group_id, mask_dataset_name, H5P_DEFAULT) <= 0) { H5Gclose(group_id); continue; }

        hid_t dataset_id = H5Dopen(group_id, mask_dataset_name, H5P_DEFAULT);
        if (dataset_id < 0) { H5Gclose(group_id); continue; }

        // Create HDF5 'memory space' to read the data into
        hid_t space_id = H5Dget_space(dataset_id);
        hid_t memspace_id = H5Screate_simple(1, (hsize_t*)&maskSize, NULL);

        // Read string attributes the helper
        std::string origin_str = readH5StringAttribute(group_id, origin_attribute_name);
        uniqueImageNames.insert(origin_str); // std::set handles uniqueness

        std::string expertise_str = readH5StringAttribute(group_id, expertise_attribute_name);
        int is_expert = (expertise_str == "Fellow" || expertise_str == "Faculty");

        // Read the actual mask data directly into its slot in the giant h_masks buffer
        checkH5Errors(H5Dread(dataset_id, H5T_NATIVE_UCHAR, memspace_id, space_id, H5P_DEFAULT,
                              h_masks + (int)numValidMasks * maskSize), "H5Dread failed");

        // Store this mask's in grouped annotations
        Annotation ann;
        ann.h_mask_index = numValidMasks; // Store its index (0, 1, 2...)
        ann.is_expert = is_expert;

        groupedAnnotations[origin_str].push_back(ann);

        MaskInfo info;
        info.imageName = origin_str;
        info.is_expert = is_expert;
        maskInfoList.push_back(info); // This list stays in 1-to-1 order with h_masks

        numValidMasks++; // Increment *after* storing the index

        H5Sclose(memspace_id); H5Sclose(space_id); H5Dclose(dataset_id); H5Gclose(group_id);
    }

    H5Fclose(file_id); // We are done with the HDF5 file
    printf("...Loaded %d valid masks from %zu unique images.\n", numValidMasks, uniqueImageNames.size());

    // --- CUDA Setup ---
    // Allocate one giant buffer on the GPU
    unsigned char* d_masks;
    size_t validMasksSize = (size_t)numValidMasks * maskSize * sizeof(unsigned char);
    CHECK(cudaMalloc(&d_masks, validMasksSize));

    // Copy all mask data from host to device in a single transfer
    CHECK(cudaMemcpy(d_masks, h_masks, validMasksSize, cudaMemcpyHostToDevice));




    // to compute time
    double startCpu, stopCpu;
    double startGpuSec, stopGpuSec;

    // Used to store final results for verification
    std::map<std::string, double> gpu_std_area_E, cpu_std_area_E;
    std::map<std::string, double> gpu_std_area_N, cpu_std_area_N;
    std::map<std::string, double> gpu_mean_entr_E, cpu_mean_entr_E;
    std::map<std::string, double> gpu_mean_entr_N, cpu_mean_entr_N;
    std::map<std::string, double> gpu_avg_dice_E, cpu_avg_dice_E;
    std::map<std::string, double> gpu_avg_dice_N, cpu_avg_dice_N;

    // 1. PARALLEL (GPU) IMPLEMENTATION ===
    printf("\n========================================\n");
    printf("Running Parallel (GPU) Implementation...\n");
    printf("========================================\n");

    // start of GPU computation
    startGpuSec = seconds();

    // GPU METRIC 1: Per-Mask Area, used for std
    // we calculate the segmented area for all masks in parallel.
    unsigned int* d_out_areas;
    CHECK(cudaMalloc(&d_out_areas, numValidMasks * sizeof(unsigned int)));
    int blockSize = 256;

    // Launch 'numValidMasks' blocks, one for each mask
    int numBlocks_Masks = numValidMasks;
    // Calculate shared memory: one 'unsigned int' per thread
    size_t sharedMemArea = (size_t)blockSize * sizeof(unsigned int);

    calculateAreaKernel<<<numBlocks_Masks, blockSize, sharedMemArea>>>(d_masks, maskSize, d_out_areas, numValidMasks);
    CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy the results (one area value per mask) back to the host
    unsigned int* h_out_areas = (unsigned int*)malloc(numValidMasks * sizeof(unsigned int));
    CHECK(cudaMemcpy(h_out_areas, d_out_areas, numValidMasks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_out_areas)); // Free GPU memory as soon as we're done

    // --- GPU METRIC 2: Mean Agreement Entropy ---
    // Multi-kernel chain. For each group (e.g., Image1-Experts):
    // 1. sumMasksKernel: Creates the probability map
    // 2. entropyMapReduceKernel: Calculates partial entropy sums
    // 3. sumReduceFloatKernel: Calculates the final total entropy sum
    printf("Calculating Mean Agreement Entropy (GPU)...\n");

    //  maps to store the final results (host-side)
    std::map<std::string, double> image_entropy_E;
    std::map<std::string, double> image_entropy_N;

    // Allocate GPU buffers that we will reuse in the loop
    unsigned int* d_sum_map; // For the probability map
    CHECK(cudaMalloc(&d_sum_map, maskSize * sizeof(unsigned int)));

    float* d_block_entropies; // For partial entropy sums
    int numBlocksEntropy = (maskSize + blockSize - 1) / blockSize;
    CHECK(cudaMalloc(&d_block_entropies, numBlocksEntropy * sizeof(float)));

    float* d_final_entropy; // For the single, final entropy sum
    CHECK(cudaMalloc(&d_final_entropy, sizeof(float)));

    size_t sharedMemFloatReduce = (size_t)blockSize * sizeof(float);

    int* d_group_mask_indices; // To hold the list of mask indices for a group

    // Loop over each image (on the CPU)
    for (std::map<std::string, std::vector<Annotation>>::const_iterator it = groupedAnnotations.begin(); it != groupedAnnotations.end(); ++it) {
        std::string imageName = it->first;
        const std::vector<Annotation>& imageAnnotations = it->second;

        // Build host-side lists of indices for this image's experts/non-experts
        std::vector<int> expert_indices;
        std::vector<int> non_expert_indices;
        for(size_t i=0; i < imageAnnotations.size(); i++) {
            if (imageAnnotations[i].is_expert) {
                expert_indices.push_back(imageAnnotations[i].h_mask_index);
            } else {
                non_expert_indices.push_back(imageAnnotations[i].h_mask_index);
            }
        }

        // --- Process EXPERT group for this image ---
        if (expert_indices.size() > 0) {
            // Copy this group's list of indices to the GPU
            CHECK(cudaMalloc(&d_group_mask_indices, expert_indices.size() * sizeof(int)));
            CHECK(cudaMemcpy(d_group_mask_indices, expert_indices.data(), expert_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

            // Kernel 1: Create probability map
            sumMasksKernel<<<numBlocksEntropy, blockSize>>>(d_masks, maskSize, d_group_mask_indices, expert_indices.size(), d_sum_map);
            // Kernel 2: Get partial entropy sums
            entropyMapReduceKernel<<<numBlocksEntropy, blockSize, sharedMemFloatReduce>>>(d_sum_map, maskSize, expert_indices.size(), d_block_entropies);
            // Kernel 3: Get final entropy sum
            sumReduceFloatKernel<<<1, blockSize, sharedMemFloatReduce>>>(d_block_entropies, numBlocksEntropy, d_final_entropy);

            // Copy the single result back
            float h_final_entropy_sum = 0.0f;
            CHECK(cudaMemcpy(&h_final_entropy_sum, d_final_entropy, sizeof(float), cudaMemcpyDeviceToHost));

            // Store the final *mean* entropy
            image_entropy_E[imageName] = (double)h_final_entropy_sum / maskSize;

            CHECK(cudaFree(d_group_mask_indices));
        }

        // --- Process NON-EXPERT group for this image ---
        // (This is the same logic as above, for the non-expert list)
        if (non_expert_indices.size() > 0) {
            CHECK(cudaMalloc(&d_group_mask_indices, non_expert_indices.size() * sizeof(int)));
            CHECK(cudaMemcpy(d_group_mask_indices, non_expert_indices.data(), non_expert_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

            sumMasksKernel<<<numBlocksEntropy, blockSize>>>(d_masks, maskSize, d_group_mask_indices, non_expert_indices.size(), d_sum_map);
            entropyMapReduceKernel<<<numBlocksEntropy, blockSize, sharedMemFloatReduce>>>(d_sum_map, maskSize, non_expert_indices.size(), d_block_entropies);
            sumReduceFloatKernel<<<1, blockSize, sharedMemFloatReduce>>>(d_block_entropies, numBlocksEntropy, d_final_entropy);

            float h_final_entropy_sum = 0.0f;
            CHECK(cudaMemcpy(&h_final_entropy_sum, d_final_entropy, sizeof(float), cudaMemcpyDeviceToHost));

            image_entropy_N[imageName] = (double)h_final_entropy_sum / maskSize;

            CHECK(cudaFree(d_group_mask_indices));
        }
    }
    // Free the re-usable GPU buffers
    CHECK(cudaFree(d_sum_map));
    CHECK(cudaFree(d_block_entropies));
    CHECK(cudaFree(d_final_entropy));

    // --- GPU METRIC 3: Per-Pair Dice Coefficient ---
    // 1. Build a single "to-do list" of all pairs (on the CPU).
    // 2. Launch ONE kernel ('diceBatchKernel') to compute all pairs.
    // 3. Copy all results back in one transfer.

    printf("Building pair list for batch processing (GPU)...\n");

    // vectors to save the valid pairs of masks.
    std::vector<int> h_pair_mask_i; // List of maskA indices
    std::vector<int> h_pair_mask_j; // List of maskB indices
    std::vector<PairInfo> h_pair_info; // List of info about each pair

    // Loop over all images, then all pairs *in that image*
    for (std::map<std::string, std::vector<Annotation>>::const_iterator it = groupedAnnotations.begin(); it != groupedAnnotations.end(); ++it) {
        std::string imageName = it->first;
        const std::vector<Annotation>& imageAnnotations = it->second;

        //  n*(n-1)/2 pairs (the if checks if the pair is valid)
        for (int i = 0; i < imageAnnotations.size(); i++) {
            for (int j = i + 1; j < imageAnnotations.size(); j++) {

                const Annotation& annA = imageAnnotations[i];
                const Annotation& annB = imageAnnotations[j];

                // We only care about EE and NN pairs
                // so if both have the same expertise
                // save the index of one in h_pair_mask_i and the other in h_pair_mask_j
                // this way the two vectors are alligned h_pair_mask_j[i] h_pair_mask_i[i] are a pair
                // then create a pair info to save which immage that pair refers to and
                // if it was expert or not, h_pair_info is also alligned to the other two vectors
                if (annA.is_expert == annB.is_expert) {
                    h_pair_mask_i.push_back(annA.h_mask_index);
                    h_pair_mask_j.push_back(annB.h_mask_index);

                    PairInfo info;
                    info.imageName = imageName;
                    info.is_expert_pair = annA.is_expert;
                    h_pair_info.push_back(info);
                }
            }
        }
    }
    int numTotalPairs = h_pair_info.size();
    printf("...Built %d total pairs (EE and NN) for Dice calculation.\n", numTotalPairs);

    // Allocate GPU memory for the "to-do list"
    int* d_pair_mask_i; int* d_pair_mask_j;
    CHECK(cudaMalloc(&d_pair_mask_i, numTotalPairs * sizeof(int)));
    CHECK(cudaMalloc(&d_pair_mask_j, numTotalPairs * sizeof(int)));

    // Allocate GPU memory for the output results
    unsigned int* d_out_intersections; unsigned int* d_out_sums;
    CHECK(cudaMalloc(&d_out_intersections, numTotalPairs * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_out_sums, numTotalPairs * sizeof(unsigned int)));

    // Copy the "to-do list" to the GPU
    CHECK(cudaMemcpy(d_pair_mask_i, h_pair_mask_i.data(), numTotalPairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pair_mask_j, h_pair_mask_j.data(), numTotalPairs * sizeof(int), cudaMemcpyHostToDevice));

    // --- LAUNCH THE KERNEL ---
    // Launch one block per pair. Each block will compute one Dice score.
    int numBlocks_Pairs = numTotalPairs;
    size_t sharedMemDice = (size_t)blockSize * 2 * sizeof(unsigned int); // intersection + sum

    diceBatchKernel<<<numBlocks_Pairs, blockSize, sharedMemDice>>>(
        d_masks, maskSize, d_pair_mask_i, d_pair_mask_j,
        d_out_intersections, d_out_sums, numTotalPairs);
    CHECK(cudaGetLastError());

    // Allocate host memory for the results
    unsigned int* h_out_intersections = (unsigned int*)malloc(numTotalPairs * sizeof(unsigned int));
    unsigned int* h_out_sums = (unsigned int*)malloc(numTotalPairs * sizeof(unsigned int));

    // Copy all results back from GPU to host in one transfer
    CHECK(cudaMemcpy(h_out_intersections, d_out_intersections, numTotalPairs * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_out_sums, d_out_sums, numTotalPairs * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Free the GPU memory for the Dice calculation
    CHECK(cudaFree(d_pair_mask_i));
    CHECK(cudaFree(d_pair_mask_j));
    CHECK(cudaFree(d_out_intersections));
    CHECK(cudaFree(d_out_sums));

    CHECK(cudaDeviceSynchronize());


    // --- GPU Post-processing and Analysis (on CPU) ---
    // Now, we are back on the CPU. We process the vectors and
    // the 'h_out' arrays we got back from the GPU to calculate the
    // final statistics.

    printf("\n--- Parallel (GPU) Implementation Results ---\n");

    // 1. Group the 'area' results by image and expertise
    std::map<std::string, std::vector<double>> image_areas_E;
    std::map<std::string, std::vector<double>> image_areas_N;
    for (int i = 0; i < numValidMasks; i++) {
        const MaskInfo& info = maskInfoList[i]; // Get info for mask 'i'
        if (info.is_expert) {
            image_areas_E[info.imageName].push_back((double)h_out_areas[i]);
        } else {
            image_areas_N[info.imageName].push_back((double)h_out_areas[i]);
        }
    }

    // 2. Group the 'Dice' results by image and expertise
    std::map<std::string, double> image_dice_sum_E;
    std::map<std::string, int> image_dice_count_E;
    std::map<std::string, double> image_dice_sum_N;
    std::map<std::string, int> image_dice_count_N;

    for (int i = 0; i < numTotalPairs; i++) {
        const PairInfo& info = h_pair_info[i];
        double dice = (2.0 * h_out_intersections[i]) / (h_out_sums[i] + epsilon);

        if (info.is_expert_pair) { // 1 = Expert
            image_dice_sum_E[info.imageName] += dice;
            image_dice_count_E[info.imageName]++;
        } else { // 0 = Non-Expert
            image_dice_sum_N[info.imageName] += dice;
            image_dice_count_N[info.imageName]++;
        }
    }

    //  print all results
    for (const std::string& imageName : uniqueImageNames) {
        printf("\nImage: \"%s\"\n", imageName.c_str());

        // --- Calculate Area & StdDev metrics ---
        double std_area_E_norm = 0.0;
        double std_area_N_norm = 0.0;
        // the areas are computed before in the GPU
        if (image_areas_E[imageName].size() > 0) {
            double l2_norm_sq = 0.0;
            for(double area : image_areas_E[imageName]) {
                l2_norm_sq += area*area;
            }
            double l2_norm = sqrt(l2_norm_sq) + epsilon;
            std::vector<double> norm_areas;
            // normalize
            for(double area : image_areas_E[imageName]) {
                norm_areas.push_back(area / l2_norm);
            }
            // find mean std sequentially
            std_area_E_norm = calculateMeanStdDev(norm_areas.data(), norm_areas.size());
        }
        // same as previous if just with non expert images
        if (image_areas_N[imageName].size() > 0) {
            double l2_norm_sq = 0.0;
            for(double area : image_areas_N[imageName]) {
                l2_norm_sq += area*area;

            }
            double l2_norm = sqrt(l2_norm_sq) + epsilon;
            std::vector<double> norm_areas;
            for(double area : image_areas_N[imageName]) {
                norm_areas.push_back(area / l2_norm);
            }
            std_area_N_norm = calculateMeanStdDev(norm_areas.data(), norm_areas.size());
        }

        // --- Get Entropy metrics ---
        double mean_entr_E = image_entropy_E[imageName];
        double mean_entr_N = image_entropy_N[imageName];

        // --- Calculate Dice metrics ---
        double avg_dice_E = (image_dice_count_E[imageName] > 0) ? image_dice_sum_E[imageName] / image_dice_count_E[imageName] : 0.0;
        double avg_dice_N = (image_dice_count_N[imageName] > 0) ? image_dice_sum_N[imageName] / image_dice_count_N[imageName] : 0.0;

        // --- Store for Verification ---
        gpu_std_area_E[imageName] = std_area_E_norm;
        gpu_std_area_N[imageName] = std_area_N_norm;
        gpu_mean_entr_E[imageName] = mean_entr_E;
        gpu_mean_entr_N[imageName] = mean_entr_N;
        gpu_avg_dice_E[imageName] = avg_dice_E;
        gpu_avg_dice_N[imageName] = avg_dice_N;

        printf("  --- Area & Normalized StdDev ---\n");
        printf("  Experts:     Norm-StdDev: %10.4f (from %zu masks)\n", std_area_E_norm, image_areas_E[imageName].size());
        printf("  Non-Experts: Norm-StdDev: %10.4f (from %zu masks)\n", std_area_N_norm, image_areas_N[imageName].size());
        printf("  --- Mean Agreement Entropy ---\n");
        printf("  Experts:     Mean Entropy: %10.4f (from %zu masks)\n", mean_entr_E, image_areas_E[imageName].size());
        printf("  Non-Experts: Mean Entropy: %10.4f (from %zu masks)\n", mean_entr_N, image_areas_N[imageName].size());
        printf("  --- Average Dice Coefficient ---\n");
        printf("  Experts:     Avg Dice: %10.4f (from %d pairs)\n", avg_dice_E, image_dice_count_E[imageName]);
        printf("  Non-Experts: Avg Dice: %10.4f (from %d pairs)\n", avg_dice_N, image_dice_count_N[imageName]);
    }
    // gpu time
    stopGpuSec = seconds();
    double gpuSeconds = stopGpuSec - startGpuSec;

        // Free host memory for GPU results
    free(h_out_intersections); free(h_out_sums); free(h_out_areas);


    printf("\n========================================\n");
    printf("Running Sequential (CPU) Implementation...\n");
    printf("========================================\n");

    startCpu = seconds();

    // --- CPU Per-Mask Metrics ---
    printf("\n--- Sequential (CPU) Implementation Results ---\n");

    std::map<std::string, std::vector<unsigned char*>> h_grouped_masks_E;
    std::map<std::string, std::vector<unsigned char*>> h_grouped_masks_N;
    for (int i = 0; i < numValidMasks; i++) {
        const MaskInfo& info = maskInfoList[i];
        // Get a pointer to the *start* of this mask in the h_masks buffer
        unsigned char* h_mask = h_masks + (long)i * maskSize;
        if (info.is_expert) {
            h_grouped_masks_E[info.imageName].push_back(h_mask);
        } else {
            h_grouped_masks_N[info.imageName].push_back(h_mask);
        }
    }

    // Calculate stats
    for (const std::string& imageName : uniqueImageNames) {
        printf("\nImage: \"%s\"\n", imageName.c_str());

        double std_area_E_norm = 0.0, mean_entr_E = 0.0, avg_dice_E = 0.0;
        double std_area_N_norm = 0.0, mean_entr_N = 0.0, avg_dice_N = 0.0;

        // Expert Results
        sequentialMaskMetrics(h_grouped_masks_E[imageName].data(),
                                h_grouped_masks_E[imageName].size(),
                                maskSize, &std_area_E_norm, &mean_entr_E, &avg_dice_E);

        // Non Expert Results
        sequentialMaskMetrics(h_grouped_masks_N[imageName].data(),
                                h_grouped_masks_N[imageName].size(),
                                maskSize, &std_area_N_norm, &mean_entr_N, &avg_dice_N);

        // Save for verification
        cpu_std_area_E[imageName] = std_area_E_norm;
        cpu_std_area_N[imageName] = std_area_N_norm;
        cpu_mean_entr_E[imageName] = mean_entr_E;
        cpu_mean_entr_N[imageName] = mean_entr_N;
        cpu_avg_dice_E[imageName] = avg_dice_E;
        cpu_avg_dice_N[imageName] = avg_dice_N;


        printf("  --- Area & Normalized StdDev ---\n");
        printf("  Experts:     Norm-StdDev: %10.4f (from %zu masks)\n", std_area_E_norm, h_grouped_masks_E[imageName].size());
        printf("  Non-Experts: Norm-StdDev: %10.4f (from %zu masks)\n", std_area_N_norm, h_grouped_masks_N[imageName].size());
        printf("  --- Mean Agreement Entropy ---\n");
        printf("  Experts:     Mean Entropy: %10.4f (from %zu masks)\n", mean_entr_E, h_grouped_masks_E[imageName].size());
        printf("  Non-Experts: Mean Entropy: %10.4f (from %zu masks)\n", mean_entr_N, h_grouped_masks_N[imageName].size());
        printf("  --- Average Dice Coefficient ---\n");
        printf("  Experts:     Avg Dice: %10.4f \n", avg_dice_E);
        printf("  Non-Experts: Avg Dice: %10.4f \n", avg_dice_N);
    }

    stopCpu = seconds();

    double cpuSeconds = (double)(stopCpu - startCpu);


    printf("\n========================================\n");
    printf("Verification (GPU vs CPU)\n");
    printf("========================================\n");

    bool all_passed = true;
    const double verification_tolerance = 1e-4; // tollerance for errors

    for (const std::string& imageName : uniqueImageNames) {
        printf("\nVerifying Image: \"%s\"\n", imageName.c_str());
        bool image_passed = true;

        auto check_metric = [&](double gpu, double cpu, const char* name) {
            bool passed = (fabs(gpu - cpu) < verification_tolerance);
            printf("  %-22s GPU: %-12.8f, CPU: %-12.8f, Pass: %s\n",
                   name, gpu, cpu, passed ? "OK" : "FAIL");
            if (!passed) image_passed = false;
        };

        check_metric(gpu_std_area_E[imageName], cpu_std_area_E[imageName], "StdDev Area (E)");
        check_metric(gpu_std_area_N[imageName], cpu_std_area_N[imageName], "StdDev Area (N)");
        check_metric(gpu_mean_entr_E[imageName], cpu_mean_entr_E[imageName], "Mean Entropy (E)");
        check_metric(gpu_mean_entr_N[imageName], cpu_mean_entr_N[imageName], "Mean Entropy (N)");
        check_metric(gpu_avg_dice_E[imageName], cpu_avg_dice_E[imageName], "Avg Dice (E)");
        check_metric(gpu_avg_dice_N[imageName], cpu_avg_dice_N[imageName], "Avg Dice (N)");

        if (!image_passed) all_passed = false;
    }

    printf("\n--- Verification Summary ---\n");
    if (all_passed) {
        printf("All metrics match! PASS\n");
    } else {
        printf("Verification FAILED! Some metrics do not match.\n");
    }

    // 4. FINAL REPORT
    printf("\n========================================\n");
    printf("Performance Report\n");
    printf("========================================\n");
    printf("Total GPU Time: %.3f s\n", gpuSeconds);
    printf("Total CPU Time: %.3f s\n", cpuSeconds);
    printf("Speedup (CPU / GPU): %.2f x\n", cpuSeconds / gpuSeconds);

    // Cleanup
    free(h_masks); // Free host memory
    CHECK(cudaFree(d_masks)); // Free device memory

    // Reset the CUDA device
    CHECK(cudaDeviceReset());

    return 0;
}


// --- KERNELS ---

/**
 * @brief CUDA KERNEL to calculate the segmented area for one mask.
 *
 * This kernel follows a "one-block-per-mask" launch strategy.
 * Each of the N blocks launched works on one of the N masks.
 *
 * Inside each block, the threads cooperate to calculate the total
 * area (sum of pixels > 0) for their assigned mask.
 *
 * ALGORITHM:
 * 1.  Grid-Stride Loop: Each of the 256 threads in the block
 * loops through the mask data (e.g., thread 0 processes pixels
 * 0, 256, 512, ...). Each thread calculates a 'local_area' sum.
 * 2. Shared Memory Reduction: The threads sum their
 * 'local_area' values together using a fast parallel reduction
 * in __shared__ memory.
 * 3. Thread 0 writes the final sum for the block to the
 * output array at the block's index (which corresponds to the mask's index).
 */
__global__ void calculateAreaKernel(const unsigned char* d_all_masks,
                                    int maskSize,
                                    unsigned int* d_out_areas,
                                    int numMasks) {
    // blockIdx.x is the index for this mask
    int maskIdx = blockIdx.x;
    if (maskIdx >= numMasks) { return; } // Safety check

    // Dynamically allocated shared memory
    extern __shared__ unsigned int s_area_reduce[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;

    // Get a pointer to the start of this block's assigned mask
    const unsigned char* d_mask = d_all_masks + (long)maskIdx * maskSize;

    // 1. Grid-Stride Loop ---
    // This block's threads (0-255) process the *entire* mask
    int globalIdx = (int)tid;
    int stride = (int)blockSize;

    unsigned int local_area = 0;
    for (int i = globalIdx; i < maskSize; i += stride) {
        if (d_mask[i] > 0) {
            local_area++;
        }
    }

    // 2. Shared Memory Reduction ---
    s_area_reduce[tid] = local_area;
    __syncthreads(); // Wait for all threads to finish map step

    // Standard "tree" reduction
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_area_reduce[tid] += s_area_reduce[tid + s];
        }
        __syncthreads(); // Sync at each step of the tree
    }

    // Thread 0 writes the final sum for this block
    if (tid == 0) {
        d_out_areas[maskIdx] = s_area_reduce[0];
    }
}

/**
 * @brief CUDA KERNEL to calculate Dice intersection/sum for a batch of pairs.
 *
 * This kernel follows a "one-block-per-pair" launch strategy.
 * Each of the N blocks launched works on one of the N pairs from the "to-do list".
 *
 * Inside each block, the threads cooperate to calculate the total
 * intersection and total pixel sum for their assigned pair.
 *
 * ALGORITHM:
 * 1. Get mask indices for this pair. Find pointers to maskA and maskB.
 * 2. Grid-Stride Loop: Each thread loops through the masks,
 * calculating *both* a 'local_intersection' and a 'local_sum'.
 * 3. Shared Memory Reduction: Threads sum their local values
 * using a parallel reduction. The shared memory is split in two
 * to handle both 'intersection' and 'sum' reductions simultaneously.
 * 4. Thread 0 writes the two final sums for the block to the
 * output arrays.
 */
__global__ void diceBatchKernel(const unsigned char* d_all_masks,
                                int maskSize,
                                const int* d_pair_mask_i,
                                const int* d_pair_mask_j,
                                unsigned int* d_out_intersections,
                                unsigned int* d_out_sums,
                                int numPairs) {
    // blockIdx.x is the index for this pair
    int pairIdx = blockIdx.x;
    if (pairIdx >= numPairs) { return; } // Safety check


    // 1, Get the global mask indices for this pair ---
    int mask_i = d_pair_mask_i[pairIdx];
    int mask_j = d_pair_mask_j[pairIdx];

    // Get pointers to the start of the two masks
    const unsigned char* d_maskA = d_all_masks + (long)mask_i * maskSize;
    const unsigned char* d_maskB = d_all_masks + (long)mask_j * maskSize;

    // Shared memory is allocated as 'blockSize * 2 * sizeof(int)'
    // We split it into two arrays:
    extern __shared__ unsigned int s_data[];
    unsigned int* s_intersection = s_data;          // First half
    unsigned int* s_sum = &s_data[blockDim.x]; // Second half

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;

    // 2. Grid-Stride Loop ---
    int globalIdx = (int)tid;
    int stride = (int)blockSize;

    unsigned int local_intersection = 0;
    unsigned int local_sum = 0;

    for (int i = globalIdx; i < maskSize; i += stride) {
        unsigned char valA = d_maskA[i] > 0 ? 1 : 0;
        unsigned char valB = d_maskB[i] > 0 ? 1 : 0;

        local_intersection += (valA & valB);
        local_sum += valA + valB;
    }

    // 3. Shared Memory Reduction ---
    s_intersection[tid] = local_intersection;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_intersection[tid] += s_intersection[tid + s];
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes both results
    if (tid == 0) {
        d_out_intersections[pairIdx] = s_intersection[0];
        d_out_sums[pairIdx] = s_sum[0];
    }
}

/**
 * @brief CUDA KERNEL to create the "probability map" (as integer sums).
 *
 * This kernel is launched with many blocks in a grid-stride
 * configuration. Each thread is responsible for one or more
 * pixels in the output 'd_out_sum_map'.
 *
 * ALGORITHM:
 * 1. Grid-Stride Loop: Each thread calculates its pixel index 'i'.
 * 2. For that pixel 'i', the thread loops through the
 * 'd_group_mask_indices' array and *gathers* the pixel value
 * from every mask in the group, summing them up.
 * 3. The thread writes this sum to 'd_out_sum_map[i]'.
 *
 * This is a "map" or "gather" operation. There is no reduction.
 */
__global__ void sumMasksKernel(const unsigned char* d_all_masks,
                               int maskSize,
                               const int* d_group_mask_indices,
                               int numMasksInGroup,
                               unsigned int* d_out_sum_map) {
    // Standard grid-stride loop setup
    int globalIdx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int stride = (int)gridDim.x * blockDim.x;

    for (int i = globalIdx; i < maskSize; i += stride) {
        unsigned int pixel_sum = 0;

        // This inner loop is a "gather"
        // Read pixel 'i' from *all* masks in this group
        for (int m = 0; m < numMasksInGroup; m++) {
            int mask_idx = d_group_mask_indices[m];
            const unsigned char* pixel_ptr = d_all_masks + (long)mask_idx * maskSize + i;

            if (*pixel_ptr > 0) {
                pixel_sum++;
            }
        }
        // Write the final sum for this pixel
        d_out_sum_map[i] = pixel_sum;
    }
}

/**
 * @brief CUDA KERNEL to calculate entropy from the sum map (Map-Reduce).
 *
 * This kernel is launched with many blocks. It takes the integer sum map
 * (from 'sumMasksKernel') and calculates the total entropy, outputting
 * a list of *partial* entropy sums (one per block).
 *
 * ALGORITHM:
 * 1. Grid-Stride Loop: Each thread calculates the entropy for
 * its assigned pixels.
 * - It reads the pixel sum 'd_sum_map[i]'.
 * - It calculates 'prob = pixel_sum / numMasks'.
 * - It calculates the entropy for that pixel.
 * - It sums these entropy values into 'local_entropy_sum'.
 * 2. Shared Memory Reduction: All threads in the block
 * sum their 'local_entropy_sum' values together.
 * 3. Thread 0 writes the block's total entropy sum to
 * 'd_block_entropies[blockIdx.x]'.
 */
__global__ void entropyMapReduceKernel(const unsigned int* d_sum_map,
                                       int maskSize,
                                       int numMasksInGroup,
                                       float* d_block_entropies) {
    // Shared memory for the reduction
    extern __shared__ float s_entropy[];

    unsigned int tid = threadIdx.x;

    // 1. Grid-Stride Loop
    int globalIdx = (int)blockIdx.x * blockDim.x + tid;
    int stride = (int)gridDim.x * blockDim.x;

    float local_entropy_sum = 0.0f;
    float fNumMasks = (float)numMasksInGroup;
    const float clip_min = 1e-8f; // Use float literal
    const float clip_max = 1.0f - clip_min;

    for (int i = globalIdx; i < maskSize; i += stride) {
        float prob = (float)d_sum_map[i] / fNumMasks;

        // Clip probability
        if (prob < clip_min) prob = clip_min;
        if (prob > clip_max) prob = clip_max;

        // Calculate entropy, adding epsilon *inside* logf
        local_entropy_sum += -(1.0f - prob) * logf((1.0f - prob) + clip_min) - prob * logf(prob + clip_min);
    }

    // 2. Shared Memory Reduction ---
    s_entropy[tid] = local_entropy_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_entropy[tid] += s_entropy[tid + s];
        }
        __syncthreads();
    }

    // write output
    if (tid == 0) {
        d_block_entropies[blockIdx.x] = s_entropy[0];
    }
}

/**
 * @brief CUDA KERNEL to perform a final reduction on a float array.
 *
 * This kernel is designed to be launched with *ONE BLOCK*.
 * It sums a potentially large array 'd_input' (e.g., the partial
 * entropy sums) into a single final value 'd_output[0]'.
 *
 * ALGORITHM:
 * 1. Grid-Stride Loop: The 256 threads in this *one block*
 * cooperatively sum all elements of 'd_input' into their
 * 'local_sum' variables.
 * 2. Shared Memory Reduction: The threads sum their
 * 'local_sum' values together.
 * 3. Thread 0 writes the single, final, grand total
 * sum to 'd_output[0]'.
 */
__global__ void sumReduceFloatKernel(const float* d_input, int N, float* d_output) {
    // Shared memory for the reduction
    extern __shared__ float s_float_data[];

    unsigned int tid = threadIdx.x;

    // 1. Grid-Stride Loop ---
    // Since gridDim.x = 1, stride is just blockDim.x
    int globalIdx = (int)blockIdx.x * blockDim.x + tid;
    int stride = (int)gridDim.x * blockDim.x;

    float local_sum = 0.0f;
    for (int i = globalIdx; i < N; i += stride) {
        local_sum += d_input[i];
    }

    // 2. Shared Memory Reduction ---
    s_float_data[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_float_data[tid] += s_float_data[tid + s];
        }
        __syncthreads();
    }

    // write output
    if (tid == 0) {
        d_output[0] = s_float_data[0];
    }
}
