// stage4_rgb_color.c - FIT ONE RGB GAUSSIAN TO REAL IMAGE PATCH
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define PATCH_SIZE 32
double g_target_patch[PATCH_SIZE * PATCH_SIZE * 3];  // RGB
int g_patch_center_x, g_patch_center_y;
double g_focal;

// Render RGB Gaussian
void render_gaussian_rgb(
    double* params,      // [6]: [x, y, z, r, g, b]
    double* out_colors   // [PATCH_SIZE * PATCH_SIZE * 3]
) {
    double x = params[0];
    double y = params[1];
    double z = params[2];
    double color_r = params[3];
    double color_g = params[4];
    double color_b = params[5];
    
    if (z < 0.1) z = 0.1;
    
    double projected_x = (x / z) * g_focal + g_patch_center_x;
    double projected_y = (y / z) * g_focal + g_patch_center_y;
    double sigma = 50.0 / z;
    
    int idx = 0;
    for (int py = 0; py < PATCH_SIZE; py++) {
        for (int px = 0; px < PATCH_SIZE; px++) {
            double pixel_x = g_patch_center_x - PATCH_SIZE/2 + px;
            double pixel_y = g_patch_center_y - PATCH_SIZE/2 + py;
            
            double dx = pixel_x - projected_x;
            double dy = pixel_y - projected_y;
            double dist_sq = dx*dx + dy*dy;
            double weight = exp(-dist_sq / (2.0 * sigma * sigma));
            
            out_colors[idx++] = color_r * weight;
            out_colors[idx++] = color_g * weight;
            out_colors[idx++] = color_b * weight;
        }
    }
}

void compute_loss_rgb(double* params, double* out_loss) {
    double rendered[PATCH_SIZE * PATCH_SIZE * 3];
    render_gaussian_rgb(params, rendered);
    
    *out_loss = 0.0;
    for (int i = 0; i < PATCH_SIZE * PATCH_SIZE * 3; i++) {
        double diff = rendered[i] - g_target_patch[i];
        *out_loss += diff * diff;
    }
    *out_loss /= (PATCH_SIZE * PATCH_SIZE * 3);
}

int main() {
    printf("=== STAGE 4: RGB Color Optimization ===\n\n");
    
    Image* img = load_png("data/r_0.png");
    if (!img) {
        printf("ERROR: Could not load data/r_0.png\n");
        return 1;
    }
    
    printf("Loaded image: %dx%d\n", img->width, img->height);
    
    g_patch_center_x = img->width / 2;
    g_patch_center_y = img->height / 2;
    g_focal = img->width / (2.0 * tan(0.6911112070083618 / 2.0));
    
    printf("Extracting %dx%d RGB patch from center\n", PATCH_SIZE, PATCH_SIZE);
    
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
    
    // Parameters: [x, y, z, r, g, b]
    double params[6] = {0.1, 0.1, 5.0, 0.5, 0.5, 0.5};  // Gray start color
    
    double loss;
    compute_loss_rgb(params, &loss);
    printf("Initial loss: %.6f\n", loss);
    printf("Initial color: (%.3f, %.3f, %.3f)\n\n", params[3], params[4], params[5]);
    
    printf("Optimizing position + color...\n");
    
    double learning_rate = 0.0001;
    int num_iters = 5000;
    
    for (int iter = 0; iter < num_iters; iter++) {
        double d_params[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        loss = 0.0;
        double d_loss = 1.0;
        
        __enzyme_autodiff(
            (void*)compute_loss_rgb,
            params, d_params,
            &loss, &d_loss
        );
        
        // Update all parameters
        for (int i = 0; i < 6; i++) {
            double grad = d_params[i];
            if (grad > 10.0) grad = 10.0;
            if (grad < -10.0) grad = -10.0;
            params[i] -= learning_rate * grad;
        }
        
        // Clamp values
        if (params[2] < 0.1) params[2] = 0.1;
        if (params[2] > 20.0) params[2] = 20.0;
        for (int i = 3; i < 6; i++) {  // Clamp colors to [0, 1]
            if (params[i] < 0.0) params[i] = 0.0;
            if (params[i] > 1.0) params[i] = 1.0;
        }
        
        if (iter % 500 == 0) {
            printf("Iter %4d: Loss=%.6f, Pos=(%.3f,%.3f,%.3f), Color=(%.3f,%.3f,%.3f)\n",
                   iter, loss, params[0], params[1], params[2],
                   params[3], params[4], params[5]);
        }
    }
    
    printf("\n=== Final Result ===\n");
    printf("Position: (%.6f, %.6f, %.6f)\n", params[0], params[1], params[2]);
    printf("Color: (%.3f, %.3f, %.3f)\n", params[3], params[4], params[5]);
    printf("Final loss: %.6f\n", loss);
    
    free(img->data);
    free(img);
    
    if (loss < 0.05) {
        printf("\n✓✓✓ SUCCESS! RGB Gaussian fits image!\n");
        printf("    Next: Add opacity and multiple Gaussians\n");
    } else {
        printf("\n⚠ Loss still high (expected for single Gaussian on complex patch)\n");
    }
    
    return 0;
}