// stage5_multiple_gaussians.c - MULTIPLE RGB GAUSSIANS WITH ALPHA BLENDING
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>

extern void __enzyme_autodiff(void*, ...);

typedef struct {
    unsigned char* data;
    int width, height;
} Image;

Image* load_png(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }
    
    png_infop info = png_create_info_struct(png);
    if (!info) { 
        png_destroy_read_struct(&png, NULL, NULL); 
        fclose(fp); 
        return NULL; 
    }
    
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }
    
    png_init_io(png, fp);
    png_read_info(png, info);
    
    Image* img = (Image*)malloc(sizeof(Image));
    img->width = png_get_image_width(png, info);
    img->height = png_get_image_height(png, info);
    
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) 
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || 
        color_type == PNG_COLOR_TYPE_GRAY || 
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || 
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    
    png_read_update_info(png, info);
    img->data = (unsigned char*)malloc(img->height * png_get_rowbytes(png, info));
    
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->height);
    for (int y = 0; y < img->height; y++) {
        row_pointers[y] = img->data + y * png_get_rowbytes(png, info);
    }
    
    png_read_image(png, row_pointers);
    png_read_end(png, info);
    
    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return img;
}

#define NUM_GAUSSIANS 10
#define PARAMS_PER_GAUSSIAN 7  // [x, y, z, r, g, b, opacity]
#define PATCH_SIZE 32

double g_target_patch[PATCH_SIZE * PATCH_SIZE * 3];
int g_patch_center_x, g_patch_center_y;
double g_focal;

// Render multiple Gaussians with alpha blending (front-to-back)
void render_multi_gaussian(
    double* params,      // [NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN]
    double* out_colors   // [PATCH_SIZE * PATCH_SIZE * 3]
) {
    // Initialize output to black
    for (int i = 0; i < PATCH_SIZE * PATCH_SIZE * 3; i++) {
        out_colors[i] = 0.0;
    }
    
    // Render each pixel
    for (int py = 0; py < PATCH_SIZE; py++) {
        for (int px = 0; px < PATCH_SIZE; px++) {
            double pixel_x = g_patch_center_x - PATCH_SIZE/2 + px;
            double pixel_y = g_patch_center_y - PATCH_SIZE/2 + py;
            
            double color_r = 0.0, color_g = 0.0, color_b = 0.0;
            double T = 1.0;  // Transmittance
            
            // Alpha composite all Gaussians (front-to-back)
            // NOTE: Should sort by depth, but skipping for simplicity
            for (int g = 0; g < NUM_GAUSSIANS; g++) {
                int base = g * PARAMS_PER_GAUSSIAN;
                double x = params[base + 0];
                double y = params[base + 1];
                double z = params[base + 2];
                double col_r = params[base + 3];
                double col_g = params[base + 4];
                double col_b = params[base + 5];
                double opacity = params[base + 6];
                
                if (z < 0.1) z = 0.1;
                
                double projected_x = (x / z) * g_focal + g_patch_center_x;
                double projected_y = (y / z) * g_focal + g_patch_center_y;
                double sigma = 50.0 / z;
                
                double dx = pixel_x - projected_x;
                double dy = pixel_y - projected_y;
                double dist_sq = dx*dx + dy*dy;
                double weight = exp(-dist_sq / (2.0 * sigma * sigma));
                
                double alpha = opacity * weight;
                if (alpha > 0.99) alpha = 0.99;  // Prevent full opacity
                
                // Alpha blending
                color_r += col_r * alpha * T;
                color_g += col_g * alpha * T;
                color_b += col_b * alpha * T;
                T *= (1.0 - alpha);
                
                if (T < 0.001) break;  // Early ray termination
            }
            
            int idx = (py * PATCH_SIZE + px) * 3;
            out_colors[idx + 0] = color_r;
            out_colors[idx + 1] = color_g;
            out_colors[idx + 2] = color_b;
        }
    }
}

void compute_loss_multi(double* params, double* out_loss) {
    double rendered[PATCH_SIZE * PATCH_SIZE * 3];
    render_multi_gaussian(params, rendered);
    
    *out_loss = 0.0;
    for (int i = 0; i < PATCH_SIZE * PATCH_SIZE * 3; i++) {
        double diff = rendered[i] - g_target_patch[i];
        *out_loss += diff * diff;
    }
    *out_loss /= (PATCH_SIZE * PATCH_SIZE * 3);
}

int main() {
    printf("=== STAGE 5: Multiple Gaussians with Alpha Blending ===\n\n");
    
    Image* img = load_png("data/r_0.png");
    if (!img) {
        printf("ERROR: Could not load data/r_0.png\n");
        return 1;
    }
    
    printf("Loaded image: %dx%d\n", img->width, img->height);
    
    g_patch_center_x = img->width / 2;
    g_patch_center_y = img->height / 2;
    g_focal = img->width / (2.0 * tan(0.6911112070083618 / 2.0));
    
    printf("Extracting %dx%d RGB patch\n", PATCH_SIZE, PATCH_SIZE);
    printf("Optimizing %d Gaussians\n\n", NUM_GAUSSIANS);
    
    // Copy RGB patch
    for (int py = 0; py < PATCH_SIZE; py++) {
        for (int px = 0; px < PATCH_SIZE; px++) {
            int img_x = g_patch_center_x - PATCH_SIZE/2 + px;
            int img_y = g_patch_center_y - PATCH_SIZE/2 + py;
            
            if (img_x >= 0 && img_x < img->width && 
                img_y >= 0 && img_y < img->height) {
                int img_idx = (img_y * img->width + img_x) * 4;
                int patch_idx = (py * PATCH_SIZE + px) * 3;
                g_target_patch[patch_idx + 0] = img->data[img_idx + 0] / 255.0;
                g_target_patch[patch_idx + 1] = img->data[img_idx + 1] / 255.0;
                g_target_patch[patch_idx + 2] = img->data[img_idx + 2] / 255.0;
            }
        }
    }
    
    // Initialize Gaussians randomly
    srand(42);
    double params[NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN];
    
    for (int g = 0; g < NUM_GAUSSIANS; g++) {
        int base = g * PARAMS_PER_GAUSSIAN;
        params[base + 0] = ((double)rand() / RAND_MAX - 0.5) * 0.2;  // x: [-0.1, 0.1]
        params[base + 1] = ((double)rand() / RAND_MAX - 0.5) * 0.2;  // y: [-0.1, 0.1]
        params[base + 2] = 4.0 + (double)rand() / RAND_MAX * 2.0;    // z: [4.0, 6.0]
        params[base + 3] = (double)rand() / RAND_MAX;                 // r
        params[base + 4] = (double)rand() / RAND_MAX;                 // g
        params[base + 5] = (double)rand() / RAND_MAX;                 // b
        params[base + 6] = 0.3 + (double)rand() / RAND_MAX * 0.4;    // opacity: [0.3, 0.7]
    }
    
    double loss;
    compute_loss_multi(params, &loss);
    printf("Initial loss: %.6f\n\n", loss);
    
    printf("Optimizing...\n");
    
    double learning_rate = 0.00005;  // Lower LR for more parameters
    int num_iters = 10000;
    
    for (int iter = 0; iter < num_iters; iter++) {
        double d_params[NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN];
        memset(d_params, 0, sizeof(d_params));
        
        loss = 0.0;
        double d_loss = 1.0;
        
        __enzyme_autodiff(
            (void*)compute_loss_multi,
            params, d_params,
            &loss, &d_loss
        );
        
        // Update all parameters
        for (int i = 0; i < NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN; i++) {
            double grad = d_params[i];
            if (grad > 10.0) grad = 10.0;
            if (grad < -10.0) grad = -10.0;
            params[i] -= learning_rate * grad;
        }
        
        // Clamp parameters
        for (int g = 0; g < NUM_GAUSSIANS; g++) {
            int base = g * PARAMS_PER_GAUSSIAN;
            
            // Clamp z
            if (params[base + 2] < 0.1) params[base + 2] = 0.1;
            if (params[base + 2] > 20.0) params[base + 2] = 20.0;
            
            // Clamp colors to [0, 1]
            for (int c = 3; c <= 5; c++) {
                if (params[base + c] < 0.0) params[base + c] = 0.0;
                if (params[base + c] > 1.0) params[base + c] = 1.0;
            }
            
            // Clamp opacity to [0, 1]
            if (params[base + 6] < 0.0) params[base + 6] = 0.0;
            if (params[base + 6] > 1.0) params[base + 6] = 1.0;
        }
        
        if (iter % 1000 == 0) {
            printf("Iter %5d: Loss=%.6f\n", iter, loss);
        }
    }
    
    printf("\n=== Final Result ===\n");
    printf("Final loss: %.6f\n\n", loss);
    
    // Show learned Gaussians
    printf("Learned Gaussians:\n");
    for (int g = 0; g < NUM_GAUSSIANS; g++) {
        int base = g * PARAMS_PER_GAUSSIAN;
        printf("  [%d] Pos=(%.3f,%.3f,%.3f) Color=(%.2f,%.2f,%.2f) Opacity=%.2f\n",
               g, params[base+0], params[base+1], params[base+2],
               params[base+3], params[base+4], params[base+5], params[base+6]);
    }
    
    free(img->data);
    free(img);
    
    if (loss < 0.03) {
        printf("\n✓✓✓ SUCCESS! Multiple Gaussians fit the image patch!\n");
        printf("    Next: Scale to full image and multiple views\n");
    } else {
        printf("\n⚠ Loss reduced but not converged. This is normal!\n");
        printf("  May need: more Gaussians, more iterations, or better initialization\n");
    }
    
    return 0;
}