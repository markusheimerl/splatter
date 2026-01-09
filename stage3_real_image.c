// stage3_real_image.c - FIT ONE GAUSSIAN TO REAL IMAGE PATCH
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>

extern void __enzyme_autodiff(void*, ...);

// Simplified image loader (based on your data.h)
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

// Global target data (so it's accessible in loss function)
#define PATCH_SIZE 32
double g_target_patch[PATCH_SIZE * PATCH_SIZE];
int g_patch_center_x, g_patch_center_y;
double g_focal;

// Render Gaussian to patch
void render_gaussian_patch(double* position, double* out_colors) {
    double z = position[2];
    if (z < 0.1) z = 0.1;
    
    double projected_x = (position[0] / z) * g_focal + g_patch_center_x;
    double projected_y = (position[1] / z) * g_focal + g_patch_center_y;
    double sigma = 50.0 / z;
    
    int idx = 0;
    for (int py = 0; py < PATCH_SIZE; py++) {
        for (int px = 0; px < PATCH_SIZE; px++) {
            double pixel_x = g_patch_center_x - PATCH_SIZE/2 + px;
            double pixel_y = g_patch_center_y - PATCH_SIZE/2 + py;
            
            double dx = pixel_x - projected_x;
            double dy = pixel_y - projected_y;
            double dist_sq = dx*dx + dy*dy;
            
            out_colors[idx++] = exp(-dist_sq / (2.0 * sigma * sigma));
        }
    }
}

void compute_loss(double* position, double* out_loss) {
    double rendered[PATCH_SIZE * PATCH_SIZE];
    render_gaussian_patch(position, rendered);
    
    *out_loss = 0.0;
    for (int i = 0; i < PATCH_SIZE * PATCH_SIZE; i++) {
        double diff = rendered[i] - g_target_patch[i];
        *out_loss += diff * diff;
    }
    *out_loss /= (PATCH_SIZE * PATCH_SIZE);
}

int main() {
    printf("=== STAGE 3: Fitting Gaussian to Real Image Patch ===\n\n");
    
    // Load first training image
    Image* img = load_png("data/r_0.png");
    if (!img) {
        printf("ERROR: Could not load data/r_0.png\n");
        printf("Make sure you're in the splatter/ directory with data/ folder\n");
        return 1;
    }
    
    printf("Loaded image: %dx%d\n", img->width, img->height);
    
    // Extract patch from image center
    g_patch_center_x = img->width / 2;
    g_patch_center_y = img->height / 2;
    g_focal = img->width / (2.0 * tan(0.6911112070083618 / 2.0));
    
    printf("Extracting %dx%d patch from center (%d, %d)\n", 
           PATCH_SIZE, PATCH_SIZE, g_patch_center_x, g_patch_center_y);
    
    // Copy patch to target buffer (convert to grayscale)
    for (int py = 0; py < PATCH_SIZE; py++) {
        for (int px = 0; px < PATCH_SIZE; px++) {
            int img_x = g_patch_center_x - PATCH_SIZE/2 + px;
            int img_y = g_patch_center_y - PATCH_SIZE/2 + py;
            
            if (img_x >= 0 && img_x < img->width && 
                img_y >= 0 && img_y < img->height) {
                int img_idx = (img_y * img->width + img_x) * 4;
                double r = img->data[img_idx + 0] / 255.0;
                double g = img->data[img_idx + 1] / 255.0;
                double b = img->data[img_idx + 2] / 255.0;
                g_target_patch[py * PATCH_SIZE + px] = (r + g + b) / 3.0;
            } else {
                g_target_patch[py * PATCH_SIZE + px] = 0.0;
            }
        }
    }
    
    // Initialize Gaussian position
    double position[3] = {0.1, 0.1, 5.0};
    
    double loss;
    compute_loss(position, &loss);
    printf("Initial loss: %.6f\n\n", loss);
    
    printf("Optimizing...\n");
    
    double learning_rate = 0.0001;
    int num_iters = 3000;
    
    for (int iter = 0; iter < num_iters; iter++) {
        double d_position[3] = {0.0, 0.0, 0.0};
        loss = 0.0;
        double d_loss = 1.0;
        
        __enzyme_autodiff(
            (void*)compute_loss,
            position, d_position,
            &loss, &d_loss
        );
        
        // Gradient descent with clipping
        for (int i = 0; i < 3; i++) {
            double grad = d_position[i];
            if (grad > 10.0) grad = 10.0;
            if (grad < -10.0) grad = -10.0;
            position[i] -= learning_rate * grad;
        }
        
        if (position[2] < 0.1) position[2] = 0.1;
        if (position[2] > 20.0) position[2] = 20.0;
        
        if (iter % 300 == 0) {
            printf("Iter %4d: Loss=%.6f, Pos=(%.4f, %.4f, %.4f)\n",
                   iter, loss, position[0], position[1], position[2]);
        }
    }
    
    printf("\n=== Final Result ===\n");
    printf("Final position: (%.6f, %.6f, %.6f)\n", position[0], position[1], position[2]);
    printf("Final loss: %.6f\n", loss);
    
    free(img->data);
    free(img);
    
    if (loss < 0.1) {
        printf("\n✓✓✓ SUCCESS! Gaussian fits real image patch!\n");
        printf("    Next: Add color, opacity, covariance optimization\n");
    } else {
        printf("\n⚠ Loss is high. Image patch may not be Gaussian-shaped.\n");
        printf("  This is expected - real images aren't perfect Gaussians!\n");
    }
    
    return 0;
}