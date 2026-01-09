// stage0_gradient_test.c - FIXED VERSION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void __enzyme_autodiff(void*, ...);

// Now Z affects the Gaussian size (perspective effect)
void evaluate_gaussian_at_pixel(
    double* position,      // [3] - Gaussian center
    double* out_color      // [1] - output color intensity
) {
    double pixel_x = 400.0;
    double pixel_y = 300.0;
    double focal = 800.0;
    
    // Perspective projection (z affects size)
    double z = position[2];
    if (z < 0.1) z = 0.1;  // Prevent division by zero
    
    double projected_x = (position[0] / z) * focal + pixel_x;
    double projected_y = (position[1] / z) * focal + pixel_y;
    
    double dx = pixel_x - projected_x;
    double dy = pixel_y - projected_y;
    
    // Gaussian size scales with depth
    double sigma = 50.0 / z;  // Closer = bigger
    double dist_sq = dx*dx + dy*dy;
    double weight = exp(-dist_sq / (2.0 * sigma * sigma));
    
    *out_color = weight;
}

int main() {
    printf("=== STAGE 0: Testing Enzyme Gradient ===\n\n");
    
    // Test 1: Slightly off-center (should have non-zero gradient)
    printf("TEST 1: Slightly off-center position\n");
    double position[3] = {0.01, 0.0, 5.0};  // MUCH smaller offset
    double d_position[3] = {0.0, 0.0, 0.0};
    double color = 0.0;
    double d_color = 1.0;
    
    __enzyme_autodiff(
        (void*)evaluate_gaussian_at_pixel,
        position, d_position,
        &color, &d_color
    );
    
    printf("Position: (%.4f, %.4f, %.4f)\n", position[0], position[1], position[2]);
    printf("Color: %.6f\n", color);
    printf("Gradient: (%.6f, %.6f, %.6f)\n", d_position[0], d_position[1], d_position[2]);
    printf("Expected: d/dx should be negative (moving right decreases color)\n");
    printf("          d/dy should be ~zero (centered in y)\n");
    printf("          d/dz should be positive (moving closer increases color)\n\n");
    
    // Test 2: Different z depth
    printf("TEST 2: Different depth\n");
    position[0] = 0.0;
    position[1] = 0.0;
    position[2] = 3.0;  // Closer
    d_position[0] = d_position[1] = d_position[2] = 0.0;
    color = 0.0;
    
    __enzyme_autodiff(
        (void*)evaluate_gaussian_at_pixel,
        position, d_position,
        &color, &d_color
    );
    
    printf("Position: (%.4f, %.4f, %.4f)\n", position[0], position[1], position[2]);
    printf("Color: %.6f\n", color);
    printf("Gradient: (%.6f, %.6f, %.6f)\n", d_position[0], d_position[1], d_position[2]);
    printf("Expected: d/dx and d/dy ~zero (centered)\n");
    printf("          d/dz should be different from Test 1\n\n");
    
    // Test 3: Verify gradient direction
    printf("TEST 3: Numerical gradient check (sanity)\n");
    position[0] = 0.01;
    position[1] = 0.0;
    position[2] = 5.0;
    
    double color1, color2;
    evaluate_gaussian_at_pixel(position, &color1);
    
    double eps = 1e-5;
    position[0] += eps;
    evaluate_gaussian_at_pixel(position, &color2);
    position[0] -= eps;
    
    double numerical_grad_x = (color2 - color1) / eps;
    
    d_position[0] = d_position[1] = d_position[2] = 0.0;
    color = 0.0;
    __enzyme_autodiff(
        (void*)evaluate_gaussian_at_pixel,
        position, d_position,
        &color, &d_color
    );
    
    printf("Numerical gradient (d/dx): %.6f\n", numerical_grad_x);
    printf("Enzyme gradient (d/dx):    %.6f\n", d_position[0]);
    printf("Relative error: %.2f%%\n", 
           fabs(numerical_grad_x - d_position[0]) / fabs(numerical_grad_x) * 100.0);
    
    if (fabs(numerical_grad_x - d_position[0]) / fabs(numerical_grad_x) < 0.01) {
        printf("\n✓✓✓ SUCCESS! Enzyme gradient matches numerical gradient!\n");
        printf("    Ready for Stage 1 (optimization).\n");
    } else {
        printf("\n✗ ERROR: Gradients don't match. Something is wrong.\n");
    }
    
    return 0;
}