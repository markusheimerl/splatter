// stage2_multiple_pixels.c - FIT ONE GAUSSIAN TO MULTIPLE PIXELS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern void __enzyme_autodiff(void*, ...);

#define NUM_PIXELS 25  // 5x5 patch around center

// Render Gaussian at multiple pixels
void render_gaussian(
    double* position,
    double* out_colors  // [NUM_PIXELS]
) {
    double focal = 800.0;
    double center_x = 400.0;
    double center_y = 300.0;
    
    double z = position[2];
    if (z < 0.1) z = 0.1;
    
    double projected_x = (position[0] / z) * focal + center_x;
    double projected_y = (position[1] / z) * focal + center_y;
    
    double sigma = 50.0 / z;
    
    // Render 5x5 patch
    int idx = 0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            double pixel_x = center_x + dx * 10.0;  // 10 pixel spacing
            double pixel_y = center_y + dy * 10.0;
            
            double diff_x = pixel_x - projected_x;
            double diff_y = pixel_y - projected_y;
            double dist_sq = diff_x*diff_x + diff_y*diff_y;
            
            out_colors[idx++] = exp(-dist_sq / (2.0 * sigma * sigma));
        }
    }
}

// Loss function: L2 difference from target pattern
void compute_loss(double* position, double* out_loss) {
    double rendered[NUM_PIXELS];
    render_gaussian(position, rendered);
    
    // Target: Bright center, darker edges (ground truth "image")
    double target[NUM_PIXELS] = {
        0.1, 0.2, 0.3, 0.2, 0.1,
        0.2, 0.5, 0.7, 0.5, 0.2,
        0.3, 0.7, 1.0, 0.7, 0.3,  // Center = 1.0
        0.2, 0.5, 0.7, 0.5, 0.2,
        0.1, 0.2, 0.3, 0.2, 0.1
    };
    
    *out_loss = 0.0;
    for (int i = 0; i < NUM_PIXELS; i++) {
        double diff = rendered[i] - target[i];
        *out_loss += diff * diff;
    }
    *out_loss /= NUM_PIXELS;  // MSE
}

int main() {
    printf("=== STAGE 2: Fitting Gaussian to 5x5 Pixel Patch ===\n\n");
    
    // Random initialization
    double position[3] = {0.12, -0.08, 4.5};
    
    printf("Target: Gaussian-shaped brightness pattern (25 pixels)\n");
    printf("Initial position: (%.4f, %.4f, %.4f)\n", position[0], position[1], position[2]);
    
    double loss = 0.0;
    compute_loss(position, &loss);
    printf("Initial loss: %.6f\n\n", loss);
    
    printf("Optimizing...\n");
    
    double learning_rate = 0.0005;
    int num_iters = 2000;
    
    for (int iter = 0; iter < num_iters; iter++) {
        double d_position[3] = {0.0, 0.0, 0.0};
        loss = 0.0;
        double d_loss = 1.0;
        
        __enzyme_autodiff(
            (void*)compute_loss,
            position, d_position,
            &loss, &d_loss
        );
        
        // Update with gradient clipping
        for (int i = 0; i < 3; i++) {
            double grad = d_position[i];
            if (grad > 10.0) grad = 10.0;
            if (grad < -10.0) grad = -10.0;
            position[i] -= learning_rate * grad;
        }
        
        if (position[2] < 0.1) position[2] = 0.1;
        if (position[2] > 20.0) position[2] = 20.0;
        
        if (iter % 200 == 0) {
            printf("Iter %4d: Loss=%.6f, Pos=(%.4f, %.4f, %.4f)\n",
                   iter, loss, position[0], position[1], position[2]);
        }
    }
    
    printf("\n=== Final Result ===\n");
    printf("Final position: (%.6f, %.6f, %.6f)\n", position[0], position[1], position[2]);
    printf("Final loss: %.6f\n\n", loss);
    
    // Show final render vs target
    double rendered[NUM_PIXELS];
    render_gaussian(position, rendered);
    
    double target[NUM_PIXELS] = {
        0.1, 0.2, 0.3, 0.2, 0.1,
        0.2, 0.5, 0.7, 0.5, 0.2,
        0.3, 0.7, 1.0, 0.7, 0.3,
        0.2, 0.5, 0.7, 0.5, 0.2,
        0.1, 0.2, 0.3, 0.2, 0.1
    };
    
    printf("Rendered vs Target (center 3x3):\n");
    for (int y = 1; y <= 3; y++) {
        for (int x = 1; x <= 3; x++) {
            int idx = y * 5 + x;
            printf("%.2f ", rendered[idx]);
        }
        printf("   |   ");
        for (int x = 1; x <= 3; x++) {
            int idx = y * 5 + x;
            printf("%.2f ", target[idx]);
        }
        printf("\n");
    }
    
    if (loss < 0.01) {
        printf("\n✓✓✓ SUCCESS! Gaussian fits the pixel pattern!\n");
        printf("    Ready for Stage 3 (load real image).\n");
    } else {
        printf("\n⚠ Loss didn't fully converge. Try:\n");
        printf("  - More iterations\n");
        printf("  - Different learning rate\n");
    }
    
    return 0;
}