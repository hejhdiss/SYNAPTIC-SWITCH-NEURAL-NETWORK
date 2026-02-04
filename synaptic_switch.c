/**
 * SYNAPTIC SWITCH - C LIBRARY
 * 
 * Implementation of "Selective Plasticity & Latent Continuity" Learning Theory
 * 
 * Core Concepts:
 * 1. Dual Weight System: Slow weights (W_p) for structure + Fast weights (W_f) for episodic memory
 * 2. Continuous Vector Latents (CVL): Stay in latent space to avoid quantization error
 * 3. Hebbian Plasticity: Fast weights updated via local Hebbian learning
 * 4. Task Vectors: Conditional activation of plasticity based on context
 * 5. Homeostatic Decay: Adaptive λ prevents hallucination and topic obsession
 * 
 * Compile to shared library:
 * Windows: gcc -shared -o synaptic_switch.dll synaptic_switch.c -lm -O3 -fopenmp   -static-libgcc -static
 * Linux:   gcc -shared -fPIC -o synaptic_switch.so synaptic_switch.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o synaptic_switch.dylib synaptic_switch.c -lm -O3 -Xpreprocessor -fopenmp -lomp
 * 
 * Licensed under GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

#define MAX_LATENT_DIM 2048
#define MIN_LATENT_DIM 64
#define DEFAULT_LATENT_DIM 512

// Homeostatic decay parameters
#define LAMBDA_MIN 0.90f
#define LAMBDA_MAX 0.999f
#define LAMBDA_DEFAULT 0.95f
#define ENERGY_THRESHOLD 10.0f  // When to increase decay

// Hebbian learning
#define HEBBIAN_RATE_DEFAULT 0.01f
#define HEBBIAN_RATE_MIN 0.001f
#define HEBBIAN_RATE_MAX 0.1f

// Task modes
typedef enum {
    TASK_REASONING = 0,  // PML bypassed, logic isolated
    TASK_CHAT = 1,       // PML active, learning enabled
    TASK_MIXED = 2       // Partial activation
} TaskMode;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    int latent_dim;           // Dimension of latent vector space
    int pml_capacity;         // Capacity of Plastic Memory Layer
    
    // === SLOW WEIGHTS (Structural Priors) - W_p ===
    // These are trained via standard backprop and remain stable
    float *W_p_encoder;       // Input → Latent [input_size × latent_dim]
    float *W_p_decoder;       // Latent → Output [latent_dim × output_size]
    float *W_p_bias_enc;      // Encoder bias [latent_dim]
    float *W_p_bias_dec;      // Decoder bias [output_size]
    
    // === FAST WEIGHTS (Synaptic Buffers) - W_f ===
    // High-volatility parameters for episodic memory
    float *W_f_keys;          // Memory keys [pml_capacity × latent_dim]
    float *W_f_values;        // Memory values [pml_capacity × latent_dim]
    float *W_f_ages;          // Age of each memory slot [pml_capacity]
    int next_slot;            // Next slot to write (circular buffer)
    
    // === STATE VECTORS ===
    float *latent_state;      // Current latent vector [latent_dim]
    float *pml_delta;         // PML personalization delta [latent_dim]
    float *output_state;      // Final output [output_size]
    
    // === TASK CONTROL ===
    TaskMode current_task;    // Current operating mode
    float pml_activation;     // 0.0 (bypass) to 1.0 (full active)
    
    // === HOMEOSTATIC DECAY ===
    float lambda;             // Current decay rate (adaptive)
    float lambda_base;        // Base decay rate
    float fast_weight_energy; // ||W_f|| norm for homeostasis
    float energy_alpha;       // EMA smoothing for energy
    
    // === LEARNING PARAMETERS ===
    float hebbian_rate;       // Learning rate for fast weights
    float slow_rate;          // Learning rate for slow weights
    float gradient_clip;      // Gradient clipping threshold
    
    // === STATISTICS ===
    int input_size;           // Original input dimension
    int output_size;          // Original output dimension
    long long update_count;   // Number of updates performed
    float avg_pml_magnitude;  // Average magnitude of PML deltas
    float avg_energy_history[100];  // Energy history for adaptive λ
    int history_idx;
    
} SynapticSwitch;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static inline float randn(void) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * M_PI * u2);
}

static inline float clip(float x, float min_val, float max_val) {
    return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
}

static inline float tanh_fast(float x) {
    return tanhf(x);
}

static inline float sigmoid_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Compute L2 norm
static float vector_norm(const float *vec, int dim) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) if(dim > 256)
    for (int i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrtf(sum);
}

// Dot product
static float dot_product(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) if(dim > 256)
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ============================================================================
// NETWORK CREATION AND DESTRUCTION
// ============================================================================

EXPORT SynapticSwitch* create_synaptic_switch(
    int input_size,
    int output_size,
    int latent_dim,
    int pml_capacity,
    float hebbian_rate,
    float slow_rate
) {
    // Validate inputs
    if (latent_dim < MIN_LATENT_DIM || latent_dim > MAX_LATENT_DIM) {
        fprintf(stderr, "Error: latent_dim must be between %d and %d\n", 
                MIN_LATENT_DIM, MAX_LATENT_DIM);
        return NULL;
    }
    
    if (pml_capacity < 1 || pml_capacity > 10000) {
        fprintf(stderr, "Error: pml_capacity must be between 1 and 10000\n");
        return NULL;
    }
    
    SynapticSwitch *net = (SynapticSwitch*)calloc(1, sizeof(SynapticSwitch));
    if (!net) return NULL;
    
    // Set dimensions
    net->input_size = input_size;
    net->output_size = output_size;
    net->latent_dim = latent_dim;
    net->pml_capacity = pml_capacity;
    
    // Allocate slow weights (W_p)
    net->W_p_encoder = (float*)malloc(input_size * latent_dim * sizeof(float));
    net->W_p_decoder = (float*)malloc(latent_dim * output_size * sizeof(float));
    net->W_p_bias_enc = (float*)calloc(latent_dim, sizeof(float));
    net->W_p_bias_dec = (float*)calloc(output_size, sizeof(float));
    
    // Allocate fast weights (W_f)
    net->W_f_keys = (float*)calloc(pml_capacity * latent_dim, sizeof(float));
    net->W_f_values = (float*)calloc(pml_capacity * latent_dim, sizeof(float));
    net->W_f_ages = (float*)calloc(pml_capacity, sizeof(float));
    
    // Allocate state vectors
    net->latent_state = (float*)calloc(latent_dim, sizeof(float));
    net->pml_delta = (float*)calloc(latent_dim, sizeof(float));
    net->output_state = (float*)calloc(output_size, sizeof(float));
    
    // Check allocations
    if (!net->W_p_encoder || !net->W_p_decoder || !net->W_f_keys || 
        !net->W_f_values || !net->latent_state) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(net);
        return NULL;
    }
    
    // Initialize parameters
    net->hebbian_rate = hebbian_rate > 0 ? hebbian_rate : HEBBIAN_RATE_DEFAULT;
    net->slow_rate = slow_rate > 0 ? slow_rate : 0.001f;
    net->gradient_clip = 5.0f;
    net->lambda_base = LAMBDA_DEFAULT;
    net->lambda = LAMBDA_DEFAULT;
    net->energy_alpha = 0.99f;
    net->current_task = TASK_CHAT;
    net->pml_activation = 1.0f;
    net->next_slot = 0;
    net->update_count = 0;
    net->fast_weight_energy = 0.0f;
    net->history_idx = 0;
    
    // Xavier initialization for slow weights
    srand(time(NULL));
    
    float scale_enc = sqrtf(2.0f / (input_size + latent_dim));
    for (int i = 0; i < input_size * latent_dim; i++) {
        net->W_p_encoder[i] = randn() * scale_enc;
    }
    
    float scale_dec = sqrtf(2.0f / (latent_dim + output_size));
    for (int i = 0; i < latent_dim * output_size; i++) {
        net->W_p_decoder[i] = randn() * scale_dec;
    }
    
    printf("✓ Synaptic Switch created: %d→[%d]→%d (PML capacity: %d)\n",
           input_size, latent_dim, output_size, pml_capacity);
    
    return net;
}

EXPORT void destroy_synaptic_switch(SynapticSwitch *net) {
    if (!net) return;
    
    free(net->W_p_encoder);
    free(net->W_p_decoder);
    free(net->W_p_bias_enc);
    free(net->W_p_bias_dec);
    free(net->W_f_keys);
    free(net->W_f_values);
    free(net->W_f_ages);
    free(net->latent_state);
    free(net->pml_delta);
    free(net->output_state);
    free(net);
}

// ============================================================================
// FORWARD PASS: Input → Latent → PML → Output
// ============================================================================

static void encode_to_latent(SynapticSwitch *net, const float *input) {
    // Compute: latent = tanh(W_p_encoder * input + bias)
    
    #pragma omp parallel for if(net->latent_dim > 128)
    for (int i = 0; i < net->latent_dim; i++) {
        float sum = net->W_p_bias_enc[i];
        for (int j = 0; j < net->input_size; j++) {
            sum += net->W_p_encoder[j * net->latent_dim + i] * input[j];
        }
        net->latent_state[i] = tanh_fast(sum);
    }
}

static void apply_pml_filter(SynapticSwitch *net) {
    // PML acts as a Dynamic Lens - adds personalization deltas
    // Only active when pml_activation > 0
    
    if (net->pml_activation < 0.001f) {
        memset(net->pml_delta, 0, net->latent_dim * sizeof(float));
        return;
    }
    
    // Compute attention scores over memory slots
    float *attention = (float*)alloca(net->pml_capacity * sizeof(float));
    float max_score = -1e9f;
    
    // Compute similarity: score[k] = latent · key[k]
    for (int k = 0; k < net->pml_capacity; k++) {
        float score = 0.0f;
        for (int i = 0; i < net->latent_dim; i++) {
            score += net->latent_state[i] * net->W_f_keys[k * net->latent_dim + i];
        }
        attention[k] = score;
        if (score > max_score) max_score = score;
    }
    
    // Softmax with temperature
    float temp = 0.1f;
    float sum_exp = 0.0f;
    for (int k = 0; k < net->pml_capacity; k++) {
        attention[k] = expf((attention[k] - max_score) / temp);
        sum_exp += attention[k];
    }
    
    if (sum_exp > 1e-10f) {
        for (int k = 0; k < net->pml_capacity; k++) {
            attention[k] /= sum_exp;
        }
    }
    
    // Compute weighted sum of values → pml_delta
    memset(net->pml_delta, 0, net->latent_dim * sizeof(float));
    
    #pragma omp parallel for if(net->latent_dim > 128)
    for (int i = 0; i < net->latent_dim; i++) {
        float delta = 0.0f;
        for (int k = 0; k < net->pml_capacity; k++) {
            delta += attention[k] * net->W_f_values[k * net->latent_dim + i];
        }
        net->pml_delta[i] = delta * net->pml_activation;
    }
    
    // Add delta to latent state
    for (int i = 0; i < net->latent_dim; i++) {
        net->latent_state[i] += net->pml_delta[i];
    }
}

static void decode_to_output(SynapticSwitch *net) {
    // Compute: output = W_p_decoder * latent + bias
    
    #pragma omp parallel for if(net->output_size > 64)
    for (int i = 0; i < net->output_size; i++) {
        float sum = net->W_p_bias_dec[i];
        for (int j = 0; j < net->latent_dim; j++) {
            sum += net->W_p_decoder[j * net->output_size + i] * net->latent_state[j];
        }
        net->output_state[i] = sum;  // Linear output (can apply activation if needed)
    }
}

EXPORT void synaptic_forward(SynapticSwitch *net, const float *input, float *output) {
    if (!net || !input || !output) return;
    
    // Step 1: Encode input to latent space
    encode_to_latent(net, input);
    
    // Step 2: Apply PML filter (if active)
    if (net->current_task == TASK_CHAT || net->current_task == TASK_MIXED) {
        apply_pml_filter(net);
    }
    
    // Step 3: Decode to output
    decode_to_output(net);
    
    // Copy output
    memcpy(output, net->output_state, net->output_size * sizeof(float));
}

// ============================================================================
// HOMEOSTATIC DECAY: Adaptive λ (forward declaration)
// ============================================================================

static void update_homeostatic_decay(SynapticSwitch *net);

// ============================================================================
// HEBBIAN UPDATE: Fast weight plasticity
// ============================================================================

EXPORT void synaptic_hebbian_update(SynapticSwitch *net, const float *input) {
    if (!net || !input) return;
    
    // Only update if in chat mode
    if (net->current_task != TASK_CHAT) return;
    
    // Forward pass to get latent state
    encode_to_latent(net, input);
    
    // Store in circular buffer
    int slot = net->next_slot;
    
    // Write key (latent state)
    memcpy(&net->W_f_keys[slot * net->latent_dim], 
           net->latent_state, 
           net->latent_dim * sizeof(float));
    
    // Write value (also latent state - can be modified for more sophistication)
    // Apply hebbian learning: Δw = η * pre * post
    for (int i = 0; i < net->latent_dim; i++) {
        float old_value = net->W_f_values[slot * net->latent_dim + i];
        float new_value = old_value * (1.0f - net->hebbian_rate) + 
                         net->hebbian_rate * net->latent_state[i];
        net->W_f_values[slot * net->latent_dim + i] = new_value;
    }
    
    // Update age
    net->W_f_ages[slot] = 0.0f;
    
    // Age all other slots and apply decay
    for (int k = 0; k < net->pml_capacity; k++) {
        if (k != slot) {
            net->W_f_ages[k] += 1.0f;
            
            // Apply lambda decay
            for (int i = 0; i < net->latent_dim; i++) {
                int idx = k * net->latent_dim + i;
                net->W_f_values[idx] *= net->lambda;
            }
        }
    }
    
    // Advance slot pointer
    net->next_slot = (net->next_slot + 1) % net->pml_capacity;
    net->update_count++;
    
    // Update homeostatic energy
    update_homeostatic_decay(net);
}

// ============================================================================
// HOMEOSTATIC DECAY: Adaptive λ
// ============================================================================

static void update_homeostatic_decay(SynapticSwitch *net) {
    // Compute energy: E = ||W_f||
    float energy = 0.0f;
    
    #pragma omp parallel for reduction(+:energy) if(net->pml_capacity > 32)
    for (int k = 0; k < net->pml_capacity; k++) {
        for (int i = 0; i < net->latent_dim; i++) {
            float val = net->W_f_values[k * net->latent_dim + i];
            energy += val * val;
        }
    }
    energy = sqrtf(energy);
    
    // EMA smoothing
    net->fast_weight_energy = net->energy_alpha * net->fast_weight_energy + 
                              (1.0f - net->energy_alpha) * energy;
    
    // Store in history
    net->avg_energy_history[net->history_idx] = net->fast_weight_energy;
    net->history_idx = (net->history_idx + 1) % 100;
    
    // Adaptive λ: increase decay if energy is too high
    if (net->fast_weight_energy > ENERGY_THRESHOLD) {
        // Cool down the network
        float excess = (net->fast_weight_energy - ENERGY_THRESHOLD) / ENERGY_THRESHOLD;
        net->lambda = net->lambda_base + (LAMBDA_MAX - net->lambda_base) * 
                      clip(excess, 0.0f, 1.0f);
    } else {
        // Return to baseline
        net->lambda = net->lambda * 0.99f + net->lambda_base * 0.01f;
    }
    
    net->lambda = clip(net->lambda, LAMBDA_MIN, LAMBDA_MAX);
}

// ============================================================================
// TASK MODE CONTROL
// ============================================================================

EXPORT void synaptic_set_task_mode(SynapticSwitch *net, int mode) {
    if (!net) return;
    
    if (mode < 0 || mode > 2) {
        fprintf(stderr, "Warning: Invalid task mode %d, using TASK_CHAT\n", mode);
        mode = TASK_CHAT;
    }
    
    net->current_task = (TaskMode)mode;
    
    switch (net->current_task) {
        case TASK_REASONING:
            net->pml_activation = 0.0f;  // PML bypassed
            break;
        case TASK_CHAT:
            net->pml_activation = 1.0f;  // PML fully active
            break;
        case TASK_MIXED:
            net->pml_activation = 0.5f;  // Partial activation
            break;
    }
}

EXPORT int synaptic_get_task_mode(SynapticSwitch *net) {
    return net ? (int)net->current_task : -1;
}

// ============================================================================
// SLOW WEIGHT TRAINING (Standard Backprop)
// ============================================================================

EXPORT float synaptic_train_slow_weights(
    SynapticSwitch *net, 
    const float *input, 
    const float *target
) {
    if (!net || !input || !target) return -1.0f;
    
    // Forward pass
    float *output = (float*)alloca(net->output_size * sizeof(float));
    synaptic_forward(net, input, output);
    
    // Compute loss (MSE)
    float loss = 0.0f;
    for (int i = 0; i < net->output_size; i++) {
        float error = output[i] - target[i];
        loss += error * error;
    }
    loss /= net->output_size;
    
    // Simple gradient descent on decoder
    // In practice, use proper backprop
    for (int i = 0; i < net->output_size; i++) {
        float error = output[i] - target[i];
        error = clip(error, -net->gradient_clip, net->gradient_clip);
        
        for (int j = 0; j < net->latent_dim; j++) {
            float grad = error * net->latent_state[j];
            net->W_p_decoder[j * net->output_size + i] -= net->slow_rate * grad;
        }
        
        net->W_p_bias_dec[i] -= net->slow_rate * error;
    }
    
    return loss;
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

EXPORT void synaptic_reset_fast_weights(SynapticSwitch *net) {
    if (!net) return;
    
    memset(net->W_f_keys, 0, net->pml_capacity * net->latent_dim * sizeof(float));
    memset(net->W_f_values, 0, net->pml_capacity * net->latent_dim * sizeof(float));
    memset(net->W_f_ages, 0, net->pml_capacity * sizeof(float));
    net->next_slot = 0;
    net->fast_weight_energy = 0.0f;
    net->lambda = net->lambda_base;
}

EXPORT void synaptic_get_latent_state(SynapticSwitch *net, float *state_out) {
    if (!net || !state_out) return;
    memcpy(state_out, net->latent_state, net->latent_dim * sizeof(float));
}

EXPORT void synaptic_get_pml_delta(SynapticSwitch *net, float *delta_out) {
    if (!net || !delta_out) return;
    memcpy(delta_out, net->pml_delta, net->latent_dim * sizeof(float));
}

// ============================================================================
// GETTERS/SETTERS
// ============================================================================

EXPORT float synaptic_get_lambda(SynapticSwitch *net) {
    return net ? net->lambda : 0.0f;
}

EXPORT void synaptic_set_lambda_base(SynapticSwitch *net, float lambda) {
    if (net) {
        net->lambda_base = clip(lambda, LAMBDA_MIN, LAMBDA_MAX);
        net->lambda = net->lambda_base;
    }
}

EXPORT float synaptic_get_energy(SynapticSwitch *net) {
    return net ? net->fast_weight_energy : 0.0f;
}

EXPORT float synaptic_get_hebbian_rate(SynapticSwitch *net) {
    return net ? net->hebbian_rate : 0.0f;
}

EXPORT void synaptic_set_hebbian_rate(SynapticSwitch *net, float rate) {
    if (net) {
        net->hebbian_rate = clip(rate, HEBBIAN_RATE_MIN, HEBBIAN_RATE_MAX);
    }
}

EXPORT long long synaptic_get_update_count(SynapticSwitch *net) {
    return net ? net->update_count : 0;
}

// ============================================================================
// STATISTICS
// ============================================================================

EXPORT void synaptic_print_stats(SynapticSwitch *net) {
    if (!net) return;
    
    // Compute statistics
    float avg_pml_mag = vector_norm(net->pml_delta, net->latent_dim);
    float avg_latent_mag = vector_norm(net->latent_state, net->latent_dim);
    
    // Count active slots
    int active_slots = 0;
    for (int k = 0; k < net->pml_capacity; k++) {
        float norm = 0.0f;
        for (int i = 0; i < net->latent_dim; i++) {
            float val = net->W_f_values[k * net->latent_dim + i];
            norm += val * val;
        }
        if (sqrtf(norm) > 0.01f) active_slots++;
    }
    
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║         SYNAPTIC SWITCH STATISTICS                    ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: %d → [%d] → %d\n", 
           net->input_size, net->latent_dim, net->output_size);
    printf("PML Capacity: %d slots\n\n", net->pml_capacity);
    
    const char *mode_str[] = {"REASONING (PML OFF)", "CHAT (PML ON)", "MIXED"};
    printf("Task Mode: %s\n", mode_str[net->current_task]);
    printf("PML Activation: %.2f%%\n\n", net->pml_activation * 100.0f);
    
    printf("Hebbian Learning:\n");
    printf("  Learning Rate:     %.4f\n", net->hebbian_rate);
    printf("  Updates:           %lld\n", net->update_count);
    printf("  Active Slots:      %d / %d\n", active_slots, net->pml_capacity);
    printf("  Next Slot:         %d\n\n", net->next_slot);
    
    printf("Homeostatic Decay:\n");
    printf("  Lambda (current):  %.4f\n", net->lambda);
    printf("  Lambda (base):     %.4f\n", net->lambda_base);
    printf("  Fast Weight Energy: %.4f\n", net->fast_weight_energy);
    printf("  Energy Threshold:  %.4f\n\n", ENERGY_THRESHOLD);
    
    printf("State Magnitudes:\n");
    printf("  Latent Vector:     %.4f\n", avg_latent_mag);
    printf("  PML Delta:         %.4f\n", avg_pml_mag);
    
    printf("\n");
}

// ============================================================================
// SAVE/LOAD
// ============================================================================

EXPORT int synaptic_save(SynapticSwitch *net, const char *filename) {
    if (!net || !filename) return -1;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    // Write header
    const char header[] = "SYNSWITCH_V1";
    fwrite(header, 1, 12, f);
    
    // Write dimensions
    fwrite(&net->input_size, sizeof(int), 1, f);
    fwrite(&net->output_size, sizeof(int), 1, f);
    fwrite(&net->latent_dim, sizeof(int), 1, f);
    fwrite(&net->pml_capacity, sizeof(int), 1, f);
    
    // Write parameters
    fwrite(&net->hebbian_rate, sizeof(float), 1, f);
    fwrite(&net->slow_rate, sizeof(float), 1, f);
    fwrite(&net->lambda_base, sizeof(float), 1, f);
    fwrite(&net->lambda, sizeof(float), 1, f);
    fwrite(&net->fast_weight_energy, sizeof(float), 1, f);
    fwrite(&net->update_count, sizeof(long long), 1, f);
    fwrite(&net->next_slot, sizeof(int), 1, f);
    
    // Write slow weights
    fwrite(net->W_p_encoder, sizeof(float), net->input_size * net->latent_dim, f);
    fwrite(net->W_p_decoder, sizeof(float), net->latent_dim * net->output_size, f);
    fwrite(net->W_p_bias_enc, sizeof(float), net->latent_dim, f);
    fwrite(net->W_p_bias_dec, sizeof(float), net->output_size, f);
    
    // Write fast weights
    fwrite(net->W_f_keys, sizeof(float), net->pml_capacity * net->latent_dim, f);
    fwrite(net->W_f_values, sizeof(float), net->pml_capacity * net->latent_dim, f);
    fwrite(net->W_f_ages, sizeof(float), net->pml_capacity, f);
    
    fclose(f);
    return 0;
}

EXPORT SynapticSwitch* synaptic_load(const char *filename) {
    if (!filename) return NULL;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    // Read and verify header
    char header[13] = {0};
    fread(header, 1, 12, f);
    if (strcmp(header, "SYNSWITCH_V1") != 0) {
        fclose(f);
        return NULL;
    }
    
    // Read dimensions
    int input_size, output_size, latent_dim, pml_capacity;
    fread(&input_size, sizeof(int), 1, f);
    fread(&output_size, sizeof(int), 1, f);
    fread(&latent_dim, sizeof(int), 1, f);
    fread(&pml_capacity, sizeof(int), 1, f);
    
    // Create network
    SynapticSwitch *net = create_synaptic_switch(
        input_size, output_size, latent_dim, pml_capacity, 0.0f, 0.0f
    );
    if (!net) {
        fclose(f);
        return NULL;
    }
    
    // Read parameters
    fread(&net->hebbian_rate, sizeof(float), 1, f);
    fread(&net->slow_rate, sizeof(float), 1, f);
    fread(&net->lambda_base, sizeof(float), 1, f);
    fread(&net->lambda, sizeof(float), 1, f);
    fread(&net->fast_weight_energy, sizeof(float), 1, f);
    fread(&net->update_count, sizeof(long long), 1, f);
    fread(&net->next_slot, sizeof(int), 1, f);
    
    // Read weights
    fread(net->W_p_encoder, sizeof(float), input_size * latent_dim, f);
    fread(net->W_p_decoder, sizeof(float), latent_dim * output_size, f);
    fread(net->W_p_bias_enc, sizeof(float), latent_dim, f);
    fread(net->W_p_bias_dec, sizeof(float), output_size, f);
    fread(net->W_f_keys, sizeof(float), pml_capacity * latent_dim, f);
    fread(net->W_f_values, sizeof(float), pml_capacity * latent_dim, f);
    fread(net->W_f_ages, sizeof(float), pml_capacity, f);
    
    fclose(f);
    return net;
}