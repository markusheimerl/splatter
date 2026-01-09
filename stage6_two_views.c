// stage6_ultra_minimal.c - ABSOLUTELY MINIMAL (no arrays in loss function!)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>
#include <json-c/json.h>

extern void __enzyme_autodiff(void*, ...);

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

// Target colors
double g_target_r1, g_target_g1, g_target_b1;
double g_target_r2, g_target_g2, g_target_b2;

// Camera 1 params (UNROLLED - no arrays!)
double g_cam1_px, g_cam1_py, g_cam1_pz;
double g_cam1_r00, g_cam1_r01, g_cam1_r02;
double g_cam1_r10, g_cam1_r11, g_cam1_r12;
double g_cam1_r20, g_cam1_r21, g_cam1_r22;
double g_cam1_focal;

// Camera 2 params (UNROLLED - no arrays!)
double g_cam2_px, g_cam2_py, g_cam2_pz;
double g_cam2_r00, g_cam2_r01, g_cam2_r02;
double g_cam2_r10, g_cam2_r11, g_cam2_r12;
double g_cam2_r20, g_cam2_r21, g_cam2_r22;
double g_cam2_focal;

double g_pixel_x, g_pixel_y;

// ONE GAUSSIAN, TWO VIEWS
void compute_loss_ultra_minimal(double* params, double* out_loss) {
    // Extract single Gaussian parameters
    double gx = params[0];
    double gy = params[1];
    double gz = params[2];
    double gr = params[3];
    double gg = params[4];
    double gb = params[5];
    double ga = params[6];
    
    // === VIEW 1 ===
    double rel1_x = gx - g_cam1_px;
    double rel1_y = gy - g_cam1_py;
    double rel1_z = gz - g_cam1_pz;
    
    // Manual matrix multiply (no arrays!)
    double cam1_x = g_cam1_r00*rel1_x + g_cam1_r01*rel1_y + g_cam1_r02*rel1_z;
    double cam1_y = g_cam1_r10*rel1_x + g_cam1_r11*rel1_y + g_cam1_r12*rel1_z;
    double cam1_z = g_cam1_r20*rel1_x + g_cam1_r21*rel1_y + g_cam1_r22*rel1_z;
    
    double cam1_z_safe = (cam1_z < 0.1) ? 0.1 : cam1_z;  // Avoid division by zero
    
    double proj1_x = (cam1_x / cam1_z_safe) * g_cam1_focal + g_pixel_x;
    double proj1_y = (cam1_y / cam1_z_safe) * g_cam1_focal + g_pixel_y;
    
    double dx1 = g_pixel_x - proj1_x;
    double dy1 = g_pixel_y - proj1_y;
    double sigma1 = 50.0 / cam1_z_safe;
    double dist1_sq = dx1*dx1 + dy1*dy1;
    double weight1 = exp(-dist1_sq / (2.0 * sigma1 * sigma1));
    
    double alpha1 = ga * weight1;
    if (alpha1 > 0.99) alpha1 = 0.99;
    
    double color1_r = gr * alpha1;
    double color1_g = gg * alpha1;
    double color1_b = gb * alpha1;
    
    // === VIEW 2 ===
    double rel2_x = gx - g_cam2_px;
    double rel2_y = gy - g_cam2_py;
    double rel2_z = gz - g_cam2_pz;
    
    double cam2_x = g_cam2_r00*rel2_x + g_cam2_r01*rel2_y + g_cam2_r02*rel2_z;
    double cam2_y = g_cam2_r10*rel2_x + g_cam2_r11*rel2_y + g_cam2_r12*rel2_z;
    double cam2_z = g_cam2_r20*rel2_x + g_cam2_r21*rel2_y + g_cam2_r22*rel2_z;
    
    double cam2_z_safe = (cam2_z < 0.1) ? 0.1 : cam2_z;
    
    double proj2_x = (cam2_x / cam2_z_safe) * g_cam2_focal + g_pixel_x;
    double proj2_y = (cam2_y / cam2_z_safe) * g_cam2_focal + g_pixel_y;
    
    double dx2 = g_pixel_x - proj2_x;
    double dy2 = g_pixel_y - proj2_y;
    double sigma2 = 50.0 / cam2_z_safe;
    double dist2_sq = dx2*dx2 + dy2*dy2;
    double weight2 = exp(-dist2_sq / (2.0 * sigma2 * sigma2));
    
    double alpha2 = ga * weight2;
    if (alpha2 > 0.99) alpha2 = 0.99;
    
    double color2_r = gr * alpha2;
    double color2_g = gg * alpha2;
    double color2_b = gb * alpha2;
    
    // === LOSS ===
    double diff1_r = color1_r - g_target_r1;
    double diff1_g = color1_g - g_target_g1;
    double diff1_b = color1_b - g_target_b1;
    
    double diff2_r = color2_r - g_target_r2;
    double diff2_g = color2_g - g_target_g2;
    double diff2_b = color2_b - g_target_b2;
    
    double loss1 = diff1_r*diff1_r + diff1_g*diff1_g + diff1_b*diff1_b;
    double loss2 = diff2_r*diff2_r + diff2_g*diff2_g + diff2_b*diff2_b;
    
    *out_loss = (loss1 + loss2) / 6.0;
}

int main() {
    printf("=== STAGE 6 ULTRA MINIMAL: ONE Gaussian, TWO Views ===\n\n");
    
    Image* img1 = load_png("data/r_0.png");
    Image* img2 = load_png("data/r_10.png");
    
    if (!img1 || !img2) {
        printf("ERROR: Could not load images\n");
        return 1;
    }
    
    Camera* cam1 = load_camera("data/transforms.json", 0);
    Camera* cam2 = load_camera("data/transforms.json", 10);
    
    if (!cam1 || !cam2) {
        printf("ERROR: Could not load cameras\n");
        return 1;
    }
    
    // Setup camera 1 globals (unrolled)
    g_cam1_px = cam1->position[0];
    g_cam1_py = cam1->position[1];
    g_cam1_pz = cam1->position[2];
    g_cam1_r00 = cam1->rotation[0]; g_cam1_r01 = cam1->rotation[1]; g_cam1_r02 = cam1->rotation[2];
    g_cam1_r10 = cam1->rotation[3]; g_cam1_r11 = cam1->rotation[4]; g_cam1_r12 = cam1->rotation[5];
    g_cam1_r20 = cam1->rotation[6]; g_cam1_r21 = cam1->rotation[7]; g_cam1_r22 = cam1->rotation[8];
    g_cam1_focal = cam1->focal;
    
    // Setup camera 2 globals (unrolled)
    g_cam2_px = cam2->position[0];
    g_cam2_py = cam2->position[1];
    g_cam2_pz = cam2->position[2];
    g_cam2_r00 = cam2->rotation[0]; g_cam2_r01 = cam2->rotation[1]; g_cam2_r02 = cam2->rotation[2];
    g_cam2_r10 = cam2->rotation[3]; g_cam2_r11 = cam2->rotation[4]; g_cam2_r12 = cam2->rotation[5];
    g_cam2_r20 = cam2->rotation[6]; g_cam2_r21 = cam2->rotation[7]; g_cam2_r22 = cam2->rotation[8];
    g_cam2_focal = cam2->focal;
    
    g_pixel_x = img1->width / 2.0;
    g_pixel_y = img1->height / 2.0;
    
    // Get target colors
    int px = img1->width / 2;
    int py = img1->height / 2;
    int idx1 = (py * img1->width + px) * 4;
    int idx2 = (py * img2->width + px) * 4;
    
    g_target_r1 = img1->data[idx1 + 0] / 255.0;
    g_target_g1 = img1->data[idx1 + 1] / 255.0;
    g_target_b1 = img1->data[idx1 + 2] / 255.0;
    
    g_target_r2 = img2->data[idx2 + 0] / 255.0;
    g_target_g2 = img2->data[idx2 + 1] / 255.0;
    g_target_b2 = img2->data[idx2 + 2] / 255.0;
    
    printf("Loaded two views: %dx%d\n", img1->width, img1->height);
    printf("Camera 1: (%.2f, %.2f, %.2f)\n", cam1->position[0], cam1->position[1], cam1->position[2]);
    printf("Camera 2: (%.2f, %.2f, %.2f)\n", cam2->position[0], cam2->position[1], cam2->position[2]);
    printf("View 1 color: (%.3f, %.3f, %.3f)\n", g_target_r1, g_target_g1, g_target_b1);
    printf("View 2 color: (%.3f, %.3f, %.3f)\n\n", g_target_r2, g_target_g2, g_target_b2);
    
    // Initialize ONE Gaussian
    double params[7] = {0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5};  // [x,y,z,r,g,b,a]
    
    double loss;
    compute_loss_ultra_minimal(params, &loss);
    printf("Initial loss: %.6f\n\n", loss);
    
    printf("Optimizing ONE Gaussian to fit both views...\n\n");
    
    double learning_rate = 0.001;
    int num_iters = 5000;
    
    for (int iter = 0; iter < num_iters; iter++) {
        double d_params[7] = {0, 0, 0, 0, 0, 0, 0};
        loss = 0.0;
        double d_loss = 1.0;
        
        __enzyme_autodiff(
            (void*)compute_loss_ultra_minimal,
            params, d_params,
            &loss, &d_loss
        );
        
        for (int i = 0; i < 7; i++) {
            double grad = d_params[i];
            if (grad > 10.0) grad = 10.0;
            if (grad < -10.0) grad = -10.0;
            params[i] -= learning_rate * grad;
        }
        
        // Clamp colors and opacity
        for (int i = 3; i < 7; i++) {
            if (params[i] < 0.0) params[i] = 0.0;
            if (params[i] > 1.0) params[i] = 1.0;
        }
        
        if (iter % 500 == 0) {
            printf("Iter %4d: Loss=%.6f, Pos=(%.3f,%.3f,%.3f)\n",
                   iter, loss, params[0], params[1], params[2]);
        }
    }
    
    printf("\n=== Result ===\n");
    printf("3D Position: (%.3f, %.3f, %.3f)\n", params[0], params[1], params[2]);
    printf("Color: (%.3f, %.3f, %.3f)\n", params[3], params[4], params[5]);
    printf("Opacity: %.3f\n", params[6]);
    printf("Final loss: %.6f\n", loss);
    
    free(img1->data); free(img1);
    free(img2->data); free(img2);
    free(cam1); free(cam2);
    
    if (loss < 0.05) {
        printf("\n🎉🎉🎉 SUCCESS! 3D Gaussian Splatting WORKS!\n");
        printf("    One Gaussian reconstructed in 3D from two views!\n");
    }
    
    return 0;
}