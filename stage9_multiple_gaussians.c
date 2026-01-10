// stage9_multiple_gaussians.c - MULTIPLE GAUSSIANS WITH ALPHA BLENDING
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>
#include <json-c/json.h>

typedef struct {
    unsigned char* data;
    int width, height;
} Image;

typedef struct {
    double position[3];
    double rotation[9];
    double focal;
} Camera;

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

Camera* load_camera(const char* json_path, int frame_idx) {
    FILE* fp = fopen(json_path, "r");
    if (!fp) return NULL;
    
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* data = (char*)malloc(length + 1);
    fread(data, 1, length, fp);
    data[length] = '\0';
    fclose(fp);
    
    json_object* root = json_tokener_parse(data);
    json_object* camera_angle_x_obj, *frames_obj;
    
    if (!json_object_object_get_ex(root, "camera_angle_x", &camera_angle_x_obj) ||
        !json_object_object_get_ex(root, "frames", &frames_obj)) {
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    double camera_angle_x = json_object_get_double(camera_angle_x_obj);
    json_object* frame = json_object_array_get_idx(frames_obj, frame_idx);
    json_object* transform_matrix_obj;
    
    if (!json_object_object_get_ex(frame, "transform_matrix", &transform_matrix_obj)) {
        free(data);
        json_object_put(root);
        return NULL;
    }
    
    Camera* cam = (Camera*)malloc(sizeof(Camera));
    
    for (int i = 0; i < 4; i++) {
        json_object* row = json_object_array_get_idx(transform_matrix_obj, i);
        for (int j = 0; j < 4; j++) {
            json_object* val = json_object_array_get_idx(row, j);
            double v = json_object_get_double(val);
            
            if (j < 3 && i < 3) {
                cam->rotation[i * 3 + j] = v;
            } else if (j == 3 && i < 3) {
                cam->position[i] = v;
            }
        }
    }
    
    cam->focal = 800.0 / (2.0 * tan(camera_angle_x / 2.0));
    
    free(data);
    json_object_put(root);
    return cam;
}

#define NUM_GAUSSIANS 5
#define PARAMS_PER_GAUSSIAN 7
#define PATCH_SIZE 128

// Forward pass: render multiple Gaussians to one pixel
void render_pixel_multi(
    double* params,      // [NUM_GAUSSIANS * 7]
    Camera* cam,
    double pixel_x, double pixel_y,
    double* out_color    // [3]
) {
    out_color[0] = 0.0;
    out_color[1] = 0.0;
    out_color[2] = 0.0;
    
    double T = 1.0;  // Transmittance
    
    for (int g = 0; g < NUM_GAUSSIANS; g++) {
        int base = g * PARAMS_PER_GAUSSIAN;
        double gx = params[base + 0];
        double gy = params[base + 1];
        double gz = params[base + 2];
        double gr = params[base + 3];
        double gg = params[base + 4];
        double gb = params[base + 5];
        double ga = params[base + 6];
        
        // Transform to camera space
        double rel_x = gx - cam->position[0];
        double rel_y = gy - cam->position[1];
        double rel_z = gz - cam->position[2];
        
        double cam_x = cam->rotation[0]*rel_x + cam->rotation[1]*rel_y + cam->rotation[2]*rel_z;
        double cam_y = cam->rotation[3]*rel_x + cam->rotation[4]*rel_y + cam->rotation[5]*rel_z;
        double cam_z = cam->rotation[6]*rel_x + cam->rotation[7]*rel_y + cam->rotation[8]*rel_z;
        
        if (cam_z < 0.1) cam_z = 0.1;
        
        // Project
        double proj_x = (cam_x / cam_z) * cam->focal + pixel_x;
        double proj_y = (cam_y / cam_z) * cam->focal + pixel_y;
        
        double dx = pixel_x - proj_x;
        double dy = pixel_y - proj_y;
        double sigma = 50.0 / cam_z;
        double dist_sq = dx*dx + dy*dy;
        double weight = exp(-dist_sq / (2.0 * sigma * sigma));
        
        double alpha = ga * weight;
        if (alpha > 0.99) alpha = 0.99;
        
        // Alpha blend
        out_color[0] += gr * alpha * T;
        out_color[1] += gg * alpha * T;
        out_color[2] += gb * alpha * T;
        
        T *= (1.0 - alpha);
        
        if (T < 0.001) break;  // Early ray termination
    }
}

// Backward pass for one pixel, multiple Gaussians
void backward_pixel_multi(
    double* params,
    Camera* cam,
    double pixel_x, double pixel_y,
    double* target_color,
    double* grad_params  // Accumulate into [NUM_GAUSSIANS * 7]
) {
    // === FORWARD PASS (save intermediates) ===
    typedef struct {
        double cam_x, cam_y, cam_z;
        double proj_x, proj_y;
        double dx, dy, sigma, dist_sq, weight, alpha;
        double gr, gg, gb;
        double T;  // Transmittance BEFORE this Gaussian
    } GaussianCache;
    
    GaussianCache cache[NUM_GAUSSIANS];
    
    double color[3] = {0, 0, 0};
    double T = 1.0;
    
    for (int g = 0; g < NUM_GAUSSIANS; g++) {
        int base = g * PARAMS_PER_GAUSSIAN;
        double gx = params[base + 0];
        double gy = params[base + 1];
        double gz = params[base + 2];
        double gr = params[base + 3];
        double gg = params[base + 4];
        double gb = params[base + 5];
        double ga = params[base + 6];
        
        double rel_x = gx - cam->position[0];
        double rel_y = gy - cam->position[1];
        double rel_z = gz - cam->position[2];
        
        double cam_x = cam->rotation[0]*rel_x + cam->rotation[1]*rel_y + cam->rotation[2]*rel_z;
        double cam_y = cam->rotation[3]*rel_x + cam->rotation[4]*rel_y + cam->rotation[5]*rel_z;
        double cam_z = cam->rotation[6]*rel_x + cam->rotation[7]*rel_y + cam->rotation[8]*rel_z;
        
        if (cam_z < 0.1) cam_z = 0.1;
        
        double proj_x = (cam_x / cam_z) * cam->focal + pixel_x;
        double proj_y = (cam_y / cam_z) * cam->focal + pixel_y;
        
        double dx = pixel_x - proj_x;
        double dy = pixel_y - proj_y;
        double sigma = 50.0 / cam_z;
        double dist_sq = dx*dx + dy*dy;
        double weight = exp(-dist_sq / (2.0 * sigma * sigma));
        
        double alpha = ga * weight;
        if (alpha > 0.99) alpha = 0.99;
        
        // Save
        cache[g].cam_x = cam_x;
        cache[g].cam_y = cam_y;
        cache[g].cam_z = cam_z;
        cache[g].proj_x = proj_x;
        cache[g].proj_y = proj_y;
        cache[g].dx = dx;
        cache[g].dy = dy;
        cache[g].sigma = sigma;
        cache[g].dist_sq = dist_sq;
        cache[g].weight = weight;
        cache[g].alpha = alpha;
        cache[g].gr = gr;
        cache[g].gg = gg;
        cache[g].gb = gb;
        cache[g].T = T;
        
        color[0] += gr * alpha * T;
        color[1] += gg * alpha * T;
        color[2] += gb * alpha * T;
        
        T *= (1.0 - alpha);
        
        if (T < 0.001) break;
    }
    
    // === BACKWARD PASS ===
    double dL_dcolor[3];
    dL_dcolor[0] = 2.0 * (color[0] - target_color[0]);
    dL_dcolor[1] = 2.0 * (color[1] - target_color[1]);
    dL_dcolor[2] = 2.0 * (color[2] - target_color[2]);
    
    double dL_dT = 0.0;  // Gradient w.r.t. transmittance (accumulated backwards)
    
    // Process Gaussians in REVERSE order (backprop through compositing)
    for (int g = NUM_GAUSSIANS - 1; g >= 0; g--) {
        int base = g * PARAMS_PER_GAUSSIAN;
        
        double alpha = cache[g].alpha;
        double T = cache[g].T;
        double gr = cache[g].gr;
        double gg = cache[g].gg;
        double gb = cache[g].gb;
        
        // Gradient from direct color contribution
        double dL_dalpha = (dL_dcolor[0] * gr + dL_dcolor[1] * gg + dL_dcolor[2] * gb) * T;
        
        // Gradient from occluding later Gaussians
        // T_next = T × (1 - alpha), so dL/dT flows back as dL/dT_next × (1 - alpha)
        // and dL/dalpha -= dL/dT_next × T
        dL_dalpha -= dL_dT * T;
        
        // Update dL_dT for next (earlier) Gaussian
        dL_dT = dL_dT * (1.0 - alpha);
        
        // Gradient w.r.t. color
        grad_params[base + 3] += dL_dcolor[0] * alpha * T;  // dL/dr
        grad_params[base + 4] += dL_dcolor[1] * alpha * T;  // dL/dg
        grad_params[base + 5] += dL_dcolor[2] * alpha * T;  // dL/db
        
        // Gradient w.r.t. weight and opacity
        double ga = params[base + 6];
        double dL_dweight = dL_dalpha * ga;
        grad_params[base + 6] += dL_dalpha * cache[g].weight;  // dL/dopacity
        
        // Gradient w.r.t. position (same as single Gaussian case)
        double sigma = cache[g].sigma;
        double weight = cache[g].weight;
        double dist_sq = cache[g].dist_sq;
        double dx = cache[g].dx;
        double dy = cache[g].dy;
        double cam_z = cache[g].cam_z;
        double cam_x = cache[g].cam_x;
        double cam_y = cache[g].cam_y;
        
        double dL_ddist_sq = dL_dweight * weight * (-1.0 / (2.0 * sigma * sigma));
        
        double dL_ddx = dL_ddist_sq * 2.0 * dx;
        double dL_ddy = dL_ddist_sq * 2.0 * dy;
        
        double dL_dproj_x = -dL_ddx;
        double dL_dproj_y = -dL_ddy;
        
        double dL_dcam_x = dL_dproj_x * (cam->focal / cam_z);
        double dL_dcam_y = dL_dproj_y * (cam->focal / cam_z);
        double dL_dcam_z = dL_dproj_x * (-cam->focal * cam_x / (cam_z * cam_z)) +
                           dL_dproj_y * (-cam->focal * cam_y / (cam_z * cam_z));
        
        double dL_dsigma = dL_dweight * weight * dist_sq / (sigma * sigma * sigma);
        dL_dcam_z += dL_dsigma * (-50.0 / (cam_z * cam_z));
        
        double dL_drel_x = cam->rotation[0]*dL_dcam_x + cam->rotation[3]*dL_dcam_y + cam->rotation[6]*dL_dcam_z;
        double dL_drel_y = cam->rotation[1]*dL_dcam_x + cam->rotation[4]*dL_dcam_y + cam->rotation[7]*dL_dcam_z;
        double dL_drel_z = cam->rotation[2]*dL_dcam_x + cam->rotation[5]*dL_dcam_y + cam->rotation[8]*dL_dcam_z;
        
        grad_params[base + 0] += dL_drel_x;
        grad_params[base + 1] += dL_drel_y;
        grad_params[base + 2] += dL_drel_z;
    }
}

int main() {
    printf("=== STAGE 9: Multiple Gaussians (%d Gaussians, %dx%d Patch) ===\n\n", 
           NUM_GAUSSIANS, PATCH_SIZE, PATCH_SIZE);
    
    Image* img = load_png("data/r_0.png");
    if (!img) {
        printf("ERROR: Could not load image\n");
        return 1;
    }
    
    Camera* cam = load_camera("data/transforms.json", 0);
    if (!cam) {
        printf("ERROR: Could not load camera\n");
        return 1;
    }
    
    int center_x = img->width / 2;
    int center_y = img->height / 2;
    
    double target[PATCH_SIZE][PATCH_SIZE][3];
    
    for (int py = 0; py < PATCH_SIZE; py++) {
        for (int px = 0; px < PATCH_SIZE; px++) {
            int img_x = center_x - PATCH_SIZE/2 + px;
            int img_y = center_y - PATCH_SIZE/2 + py;
            int idx = (img_y * img->width + img_x) * 4;
            
            target[py][px][0] = img->data[idx + 0] / 255.0;
            target[py][px][1] = img->data[idx + 1] / 255.0;
            target[py][px][2] = img->data[idx + 2] / 255.0;
        }
    }
    
    printf("Loaded %dx%d patch\n", PATCH_SIZE, PATCH_SIZE);
    printf("Optimizing %d Gaussians (%" PRIu64 " parameters total)\n\n", 
           NUM_GAUSSIANS, (uint64_t)(NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN));
    
    // Initialize Gaussians randomly
    srand(42);
    double params[NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN];
    
    for (int g = 0; g < NUM_GAUSSIANS; g++) {
        int base = g * PARAMS_PER_GAUSSIAN;
        params[base + 0] = ((double)rand() / RAND_MAX - 0.5) * 0.4;  // x
        params[base + 1] = ((double)rand() / RAND_MAX - 0.5) * 0.4;  // y
        params[base + 2] = ((double)rand() / RAND_MAX - 0.5) * 0.4;  // z
        params[base + 3] = (double)rand() / RAND_MAX;                // r
        params[base + 4] = (double)rand() / RAND_MAX;                // g
        params[base + 5] = (double)rand() / RAND_MAX;                // b
        params[base + 6] = 0.3 + (double)rand() / RAND_MAX * 0.4;   // opacity
    }
    
    double learning_rate = 0.0005 / (PATCH_SIZE * PATCH_SIZE);
    int num_iters = 10000;
    
    printf("Optimizing...\n\n");
    
    for (int iter = 0; iter < num_iters; iter++) {
        double grad[NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN];
        memset(grad, 0, sizeof(grad));
        
        // Accumulate gradients from all pixels
        for (int py = 0; py < PATCH_SIZE; py++) {
            for (int px = 0; px < PATCH_SIZE; px++) {
                double pixel_x = center_x - PATCH_SIZE/2 + px;
                double pixel_y = center_y - PATCH_SIZE/2 + py;
                
                backward_pixel_multi(params, cam, pixel_x, pixel_y, target[py][px], grad);
            }
        }
        
        // Gradient descent
        for (int i = 0; i < NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN; i++) {
            params[i] -= learning_rate * grad[i];
        }
        
        // Clamp colors and opacity
        for (int g = 0; g < NUM_GAUSSIANS; g++) {
            int base = g * PARAMS_PER_GAUSSIAN;
            for (int c = 3; c <= 6; c++) {
                if (params[base + c] < 0.0) params[base + c] = 0.0;
                if (params[base + c] > 1.0) params[base + c] = 1.0;
            }
        }
        
        if (iter % 1000 == 0) {
            double loss = 0.0;
            
            for (int py = 0; py < PATCH_SIZE; py++) {
                for (int px = 0; px < PATCH_SIZE; px++) {
                    double pixel_x = center_x - PATCH_SIZE/2 + px;
                    double pixel_y = center_y - PATCH_SIZE/2 + py;
                    double color[3];
                    
                    render_pixel_multi(params, cam, pixel_x, pixel_y, color);
                    
                    for (int c = 0; c < 3; c++) {
                        double diff = color[c] - target[py][px][c];
                        loss += diff * diff;
                    }
                }
            }
            
            loss /= (PATCH_SIZE * PATCH_SIZE * 3);
            
            printf("Iter %5d: Loss=%.6f\n", iter, loss);
        }
    }
    
    printf("\n=== Final Gaussians ===\n");
    for (int g = 0; g < NUM_GAUSSIANS; g++) {
        int base = g * PARAMS_PER_GAUSSIAN;
        printf("[%d] Pos=(%.3f,%.3f,%.3f) Color=(%.2f,%.2f,%.2f) Opacity=%.2f\n",
               g, params[base+0], params[base+1], params[base+2],
               params[base+3], params[base+4], params[base+5], params[base+6]);
    }
    
    // Test center pixel
    double center_color[3];
    render_pixel_multi(params, cam, center_x, center_y, center_color);
    printf("\nCenter pixel:\n");
    printf("  Rendered: (%.3f, %.3f, %.3f)\n", center_color[0], center_color[1], center_color[2]);
    printf("  Target:   (%.3f, %.3f, %.3f)\n", 
           target[PATCH_SIZE/2][PATCH_SIZE/2][0],
           target[PATCH_SIZE/2][PATCH_SIZE/2][1],
           target[PATCH_SIZE/2][PATCH_SIZE/2][2]);
    
    free(img->data);
    free(img);
    free(cam);
    
    printf("\n✓✓✓ SUCCESS! Multiple Gaussians with alpha blending!\n");
    printf("    Ready for Stage 10 (full resolution + more Gaussians)\n");
    
    return 0;
}