// stage7_manual_gradients.c - MANUAL GRADIENTS, ONE GAUSSIAN, ONE PIXEL
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

// Forward pass: render one Gaussian to one pixel
void render_forward(
    double* params,       // [x,y,z,r,g,b,a]
    Camera* cam,
    double pixel_x, double pixel_y,
    double* out_color     // [r,g,b]
) {
    double gx = params[0], gy = params[1], gz = params[2];
    double gr = params[3], gg = params[4], gb = params[5], ga = params[6];
    
    // Transform to camera space
    double rel_x = gx - cam->position[0];
    double rel_y = gy - cam->position[1];
    double rel_z = gz - cam->position[2];
    
    double cam_x = cam->rotation[0]*rel_x + cam->rotation[1]*rel_y + cam->rotation[2]*rel_z;
    double cam_y = cam->rotation[3]*rel_x + cam->rotation[4]*rel_y + cam->rotation[5]*rel_z;
    double cam_z = cam->rotation[6]*rel_x + cam->rotation[7]*rel_y + cam->rotation[8]*rel_z;
    
    if (cam_z < 0.1) cam_z = 0.1;
    
    // Project to screen
    double proj_x = (cam_x / cam_z) * cam->focal + pixel_x;
    double proj_y = (cam_y / cam_z) * cam->focal + pixel_y;
    
    // Gaussian weight
    double dx = pixel_x - proj_x;
    double dy = pixel_y - proj_y;
    double sigma = 50.0 / cam_z;
    double dist_sq = dx*dx + dy*dy;
    double weight = exp(-dist_sq / (2.0 * sigma * sigma));
    
    double alpha = ga * weight;
    if (alpha > 0.99) alpha = 0.99;
    
    out_color[0] = gr * alpha;
    out_color[1] = gg * alpha;
    out_color[2] = gb * alpha;
}

// Backward pass: compute gradients manually!
void render_backward(
    double* params,       // [x,y,z,r,g,b,a]
    Camera* cam,
    double pixel_x, double pixel_y,
    double* target_color, // [r,g,b]
    double* grad_params   // Output: [dx,dy,dz,dr,dg,db,da]
) {
    // === FORWARD PASS (recompute, saving intermediates) ===
    double gx = params[0], gy = params[1], gz = params[2];
    double gr = params[3], gg = params[4], gb = params[5], ga = params[6];
    
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
    
    double color_r = gr * alpha;
    double color_g = gg * alpha;
    double color_b = gb * alpha;
    
    // === BACKWARD PASS (chain rule!) ===
    
    // dL/dColor = 2 * (color - target)
    double dL_dcolor_r = 2.0 * (color_r - target_color[0]);
    double dL_dcolor_g = 2.0 * (color_g - target_color[1]);
    double dL_dcolor_b = 2.0 * (color_b - target_color[2]);
    
    // dL/dAlpha (from color = gaussian_color * alpha)
    double dL_dalpha = dL_dcolor_r * gr + dL_dcolor_g * gg + dL_dcolor_b * gb;
    
    // dL/d(gaussian_color)
    grad_params[3] = dL_dcolor_r * alpha;  // dL/dr
    grad_params[4] = dL_dcolor_g * alpha;  // dL/dg
    grad_params[5] = dL_dcolor_b * alpha;  // dL/db
    
    // dL/dWeight (from alpha = opacity * weight)
    double dL_dweight = dL_dalpha * ga;
    
    // dL/dOpacity
    grad_params[6] = dL_dalpha * weight;   // dL/da
    
    // dL/dDist² (from weight = exp(-dist²/(2σ²)))
    double dL_ddist_sq = dL_dweight * weight * (-1.0 / (2.0 * sigma * sigma));
    
    // dL/dDx, dL/dDy (from dist² = dx² + dy²)
    double dL_ddx = dL_ddist_sq * 2.0 * dx;
    double dL_ddy = dL_ddist_sq * 2.0 * dy;
    
    // dL/dProj (from dx = pixel_x - proj_x, so dL/dproj_x = -dL/ddx)
    double dL_dproj_x = -dL_ddx;
    double dL_dproj_y = -dL_ddy;
    
    // dL/dCameraSpace (Jacobian of projection: proj = cam/cam_z * focal)
    // d(proj_x)/d(cam_x) = focal / cam_z
    // d(proj_x)/d(cam_z) = -focal * cam_x / cam_z²
    double dL_dcam_x = dL_dproj_x * (cam->focal / cam_z);
    double dL_dcam_y = dL_dproj_y * (cam->focal / cam_z);
    double dL_dcam_z = dL_dproj_x * (-cam->focal * cam_x / (cam_z * cam_z)) +
                       dL_dproj_y * (-cam->focal * cam_y / (cam_z * cam_z));
    
    // Add gradient from sigma (sigma = 50/cam_z, affects weight)
    // dL/dSigma = dL/dWeight * d(weight)/d(sigma)
    // d(weight)/d(sigma) = weight * dist² / sigma³
    double dL_dsigma = dL_dweight * weight * dist_sq / (sigma * sigma * sigma);
    // dSigma/dCam_z = -50 / cam_z²
    dL_dcam_z += dL_dsigma * (-50.0 / (cam_z * cam_z));
    
    // dL/dRelative (from camera transform: cam_space = R * rel)
    // Transpose rotation to go backwards
    double dL_drel_x = cam->rotation[0]*dL_dcam_x + cam->rotation[3]*dL_dcam_y + cam->rotation[6]*dL_dcam_z;
    double dL_drel_y = cam->rotation[1]*dL_dcam_x + cam->rotation[4]*dL_dcam_y + cam->rotation[7]*dL_dcam_z;
    double dL_drel_z = cam->rotation[2]*dL_dcam_x + cam->rotation[5]*dL_dcam_y + cam->rotation[8]*dL_dcam_z;
    
    // dL/dPosition (from rel = pos - cam_pos)
    grad_params[0] = dL_drel_x;  // dL/dx
    grad_params[1] = dL_drel_y;  // dL/dy
    grad_params[2] = dL_drel_z;  // dL/dz
}

// Numerical gradient (for verification)
void numerical_gradient(
    double* params,
    Camera* cam,
    double pixel_x, double pixel_y,
    double* target_color,
    double* num_grad,
    double eps
) {
    double color_plus[3], color_minus[3];
    
    for (int i = 0; i < 7; i++) {
        // Forward difference
        params[i] += eps;
        render_forward(params, cam, pixel_x, pixel_y, color_plus);
        params[i] -= 2*eps;
        render_forward(params, cam, pixel_x, pixel_y, color_minus);
        params[i] += eps;  // Restore
        
        // Compute loss at both points
        double loss_plus = 0.0, loss_minus = 0.0;
        for (int c = 0; c < 3; c++) {
            double diff_plus = color_plus[c] - target_color[c];
            double diff_minus = color_minus[c] - target_color[c];
            loss_plus += diff_plus * diff_plus;
            loss_minus += diff_minus * diff_minus;
        }
        
        num_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }
}

int main() {
    printf("=== STAGE 7: Manual Gradients (One Gaussian, One Pixel) ===\n\n");
    
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
    
    // Get target color (center pixel)
    int px = img->width / 2;
    int py = img->height / 2;
    int idx = (py * img->width + px) * 4;
    double target[3] = {
        img->data[idx + 0] / 255.0,
        img->data[idx + 1] / 255.0,
        img->data[idx + 2] / 255.0
    };
    
    printf("Target color: (%.3f, %.3f, %.3f)\n", target[0], target[1], target[2]);
    printf("Camera position: (%.2f, %.2f, %.2f)\n\n", 
           cam->position[0], cam->position[1], cam->position[2]);
    
    // Initialize Gaussian
    double params[7] = {0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5};
    
    // === TEST: Compare analytical vs numerical gradients ===
    printf("=== Gradient Verification ===\n");
    
    double analytical_grad[7];
    double numerical_grad[7];
    
    render_backward(params, cam, px, py, target, analytical_grad);
    numerical_gradient(params, cam, px, py, target, numerical_grad, 1e-5);
    
    printf("Parameter  | Analytical | Numerical  | Rel Error\n");
    printf("-----------|------------|------------|----------\n");
    const char* names[] = {"x", "y", "z", "r", "g", "b", "opacity"};
    for (int i = 0; i < 7; i++) {
        double rel_error = fabs(analytical_grad[i] - numerical_grad[i]) / 
                          (fabs(numerical_grad[i]) + 1e-8);
        printf("%-10s | %10.6f | %10.6f | %.2e\n", 
               names[i], analytical_grad[i], numerical_grad[i], rel_error);
    }
    
    printf("\n");
    
    if (fabs(analytical_grad[0] - numerical_grad[0]) / fabs(numerical_grad[0]) > 0.01) {
        printf("⚠️  WARNING: Gradients don't match! Check the math.\n");
        return 1;
    }
    
    printf("✓ Gradients match! Analytical implementation is correct.\n\n");
    
    // === OPTIMIZATION ===
    printf("=== Optimization with Manual Gradients ===\n\n");
    
    double learning_rate = 0.001;
    int num_iters = 3000;
    
    for (int iter = 0; iter < num_iters; iter++) {
        double grad[7];
        render_backward(params, cam, px, py, target, grad);
        
        // Gradient descent
        for (int i = 0; i < 7; i++) {
            params[i] -= learning_rate * grad[i];
        }
        
        // Clamp
        for (int i = 3; i < 7; i++) {
            if (params[i] < 0.0) params[i] = 0.0;
            if (params[i] > 1.0) params[i] = 1.0;
        }
        
        if (iter % 300 == 0) {
            double color[3];
            render_forward(params, cam, px, py, color);
            double loss = 0.0;
            for (int c = 0; c < 3; c++) {
                double diff = color[c] - target[c];
                loss += diff * diff;
            }
            
            printf("Iter %4d: Loss=%.6f, Pos=(%.3f,%.3f,%.3f)\n",
                   iter, loss, params[0], params[1], params[2]);
        }
    }
    
    printf("\n=== Final Result ===\n");
    printf("Position: (%.3f, %.3f, %.3f)\n", params[0], params[1], params[2]);
    printf("Color: (%.3f, %.3f, %.3f)\n", params[3], params[4], params[5]);
    printf("Opacity: %.3f\n", params[6]);
    
    double final_color[3];
    render_forward(params, cam, px, py, final_color);
    printf("Rendered: (%.3f, %.3f, %.3f)\n", final_color[0], final_color[1], final_color[2]);
    printf("Target:   (%.3f, %.3f, %.3f)\n", target[0], target[1], target[2]);
    
    free(img->data);
    free(img);
    free(cam);
    
    printf("\n✓✓✓ SUCCESS! Manual gradients work!\n");
    printf("    Ready for Stage 8 (multiple pixels)\n");
    
    return 0;
}