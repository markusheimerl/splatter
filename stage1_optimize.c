// stage1_optimize.c - FIT ONE GAUSSIAN TO ONE PIXEL
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void __enzyme_autodiff(void*, ...);

// Same rendering function
void evaluate_gaussian_at_pixel(
    double* position,
    double* out_color
) {
    double pixel_x = 400.0;
    double pixel_y = 300.0;
    double focal = 800.0;
    
    double z = position[2];
    if (z < 0.1) z = 0.1;
    
    double projected_x = (position[0] / z) * focal + pixel_x;
    double projected_y = (position[1] / z) * focal + pixel_y;
    
    double dx = pixel_x - projected_x;
    double dy = pixel_y - projected_y;
    
    double sigma = 50.0 / z;
    double dist_sq = dx*dx + dy*dy;
    double weight = exp(-dist_sq / (2.0 * sigma * sigma));
    
    *out_color = weight;
}

// Loss function wrapper (like your quadcopter!)
void compute_loss(double* position, double* out_loss) {
    double color = 0.0;
    evaluate_gaussian_at_pixel(position, &color);
    
    double target = 0.5;  // Target: 50% brightness at center pixel
    double diff = color - target;
    *out_loss = diff * diff;  // L2 loss
}

int main() {
    printf("=== STAGE 1: Optimizing Gaussian Position ===\n\n");
    
    // Start Gaussian in a bad position
    double position[3] = {0.05, 0.08, 5.0};  // Off-center
    
    printf("Initial position: (%.4f, %.4f, %.4f)\n", position[0], position[1], position[2]);
    
    double color = 0.0;
    evaluate_gaussian_at_pixel(position, &color);
    printf("Initial color at pixel: %.6f (target: 0.500000)\n", color);
    printf("Initial loss: %.6f\n\n", (color - 0.5) * (color - 0.5));
    
    // Optimization parameters
    double learning_rate = 0.001;
    int num_iters = 1000;
    
    printf("Starting optimization...\n");
    
    for (int iter = 0; iter < num_iters; iter++) {
        // Compute loss and gradient
        double d_position[3] = {0.0, 0.0, 0.0};
        double loss = 0.0;
        double d_loss = 1.0;
        
        __enzyme_autodiff(
            (void*)compute_loss,
            position, d_position,
            &loss, &d_loss
        );
        
        // Gradient descent update
        for (int i = 0; i < 3; i++) {
            position[i] -= learning_rate * d_position[i];
        }
        
        // Clamp z to positive values
        if (position[2] < 0.1) position[2] = 0.1;
        
        if (iter % 100 == 0) {
            printf("Iter %4d: Loss=%.6f, Pos=(%.4f, %.4f, %.4f), Grad_norm=%.4f\n",
                   iter, loss, position[0], position[1], position[2],
                   sqrt(d_position[0]*d_position[0] + d_position[1]*d_position[1] + d_position[2]*d_position[2]));
        }
    }
    
    printf("\n=== Final Result ===\n");
    printf("Final position: (%.6f, %.6f, %.6f)\n", position[0], position[1], position[2]);
    
    color = 0.0;
    evaluate_gaussian_at_pixel(position, &color);
    printf("Final color: %.6f (target: 0.500000)\n", color);
    printf("Final error: %.6f\n", fabs(color - 0.5));
    
    printf("\nExpected: Position should be close to (0, 0, ~6.3)\n");
    printf("          (x=0, y=0 centers it, z adjusted to get 50%% brightness)\n");
    
    if (fabs(position[0]) < 0.01 && fabs(position[1]) < 0.01 && fabs(color - 0.5) < 0.01) {
        printf("\n✓✓✓ SUCCESS! Gaussian optimized to target pixel!\n");
        printf("    Ready for Stage 2 (multiple pixels).\n");
    } else {
        printf("\n⚠ Loss decreased but didn't fully converge. May need:\n");
        printf("  - Lower learning rate\n");
        printf("  - More iterations\n");
        printf("  - Better initialization\n");
    }
    
    return 0;
}