#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <webp/encode.h>

#define WIDTH 800
#define HEIGHT 600
#define NUM_GAUSSIANS 1000

// 3D vector
typedef struct {
    float x, y, z;
} vec3;

// 3x3 matrix
typedef struct {
    float m[3][3];
} mat3;

// 3D Gaussian splat
typedef struct {
    vec3 position;
    mat3 covariance;
    float color[3];  // RGB
    float opacity;
} Gaussian;

// Camera
typedef struct {
    vec3 position;
    vec3 forward;
    vec3 up;
    vec3 right;
    float fov;
} Camera;

// Projected 2D Gaussian
typedef struct {
    float x, y;
    float cov[3];    // 2x2 covariance [a, b, c] for [[a,b],[b,c]]
    float color[3];
    float opacity;
    float depth;
    int valid;
} Gaussian2D;

// Utility functions
float randf(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

vec3 vec3_sub(vec3 a, vec3 b) {
    return (vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 vec3_scale(vec3 v, float s) {
    return (vec3){v.x * s, v.y * s, v.z * s};
}

float vec3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 vec3_cross(vec3 a, vec3 b) {
    return (vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

vec3 vec3_normalize(vec3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-6f) return (vec3){0, 0, 1};
    return vec3_scale(v, 1.0f / len);
}

mat3 mat3_mul(mat3 a, mat3 b) {
    mat3 result = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return result;
}

mat3 mat3_transpose(mat3 m) {
    mat3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = m.m[j][i];
        }
    }
    return result;
}

// Create random covariance matrix (positive semi-definite)
mat3 random_covariance() {
    float scale_x = randf(0.05f, 0.3f);
    float scale_y = randf(0.05f, 0.3f);
    float scale_z = randf(0.05f, 0.3f);
    
    float rx = randf(0, 2 * M_PI);
    float ry = randf(0, 2 * M_PI);
    float rz = randf(0, 2 * M_PI);
    
    // Rotation matrix from Euler angles
    float cx = cosf(rx), sx = sinf(rx);
    float cy = cosf(ry), sy = sinf(ry);
    float cz = cosf(rz), sz = sinf(rz);
    
    mat3 rot;
    rot.m[0][0] = cy * cz;
    rot.m[0][1] = -cy * sz;
    rot.m[0][2] = sy;
    rot.m[1][0] = sx * sy * cz + cx * sz;
    rot.m[1][1] = -sx * sy * sz + cx * cz;
    rot.m[1][2] = -sx * cy;
    rot.m[2][0] = -cx * sy * cz + sx * sz;
    rot.m[2][1] = cx * sy * sz + sx * cz;
    rot.m[2][2] = cx * cy;
    
    // Scale matrix
    mat3 scale = {0};
    scale.m[0][0] = scale_x * scale_x;
    scale.m[1][1] = scale_y * scale_y;
    scale.m[2][2] = scale_z * scale_z;
    
    // Covariance = R * S * R^T
    mat3 temp = mat3_mul(rot, scale);
    return mat3_mul(temp, mat3_transpose(rot));
}

// Initialize k random Gaussians
void init_gaussians(Gaussian *gaussians, int count) {
    for (int i = 0; i < count; i++) {
        gaussians[i].position = (vec3){
            randf(-5.0f, 5.0f),
            randf(-5.0f, 5.0f),
            randf(-5.0f, 5.0f)
        };
        
        gaussians[i].covariance = random_covariance();
        
        gaussians[i].color[0] = randf(0.0f, 1.0f);
        gaussians[i].color[1] = randf(0.0f, 1.0f);
        gaussians[i].color[2] = randf(0.0f, 1.0f);
        
        gaussians[i].opacity = randf(0.3f, 0.9f);
    }
}

// Create camera at random position looking at origin
Camera create_random_camera() {
    Camera cam;
    
    float theta = randf(0, 2 * M_PI);
    float phi = randf(-M_PI / 6, M_PI / 6);
    float radius = randf(8.0f, 12.0f);
    
    cam.position = (vec3){
        radius * cosf(phi) * cosf(theta),
        radius * sinf(phi),
        radius * cosf(phi) * sinf(theta)
    };
    
    vec3 target = {0, 0, 0};
    cam.forward = vec3_normalize(vec3_sub(target, cam.position));
    
    vec3 world_up = {0, 1, 0};
    cam.right = vec3_normalize(vec3_cross(cam.forward, world_up));
    cam.up = vec3_cross(cam.right, cam.forward);
    
    cam.fov = 60.0f * M_PI / 180.0f;
    
    return cam;
}

// Project 3D Gaussian to 2D screen space
Gaussian2D project_gaussian(Gaussian *g, Camera *cam) {
    Gaussian2D g2d = {0};
    
    // Transform to camera space
    vec3 pos_cam = vec3_sub(g->position, cam->position);
    float x = vec3_dot(pos_cam, cam->right);
    float y = vec3_dot(pos_cam, cam->up);
    float z = vec3_dot(pos_cam, cam->forward);
    
    // Cull if behind camera
    if (z < 0.1f) {
        g2d.valid = 0;
        return g2d;
    }
    
    g2d.depth = z;
    
    // Perspective projection
    float focal = HEIGHT / (2.0f * tanf(cam->fov / 2.0f));
    g2d.x = (x / z) * focal + WIDTH / 2.0f;
    g2d.y = (-y / z) * focal + HEIGHT / 2.0f;
    
    // Transform covariance to camera space
    mat3 view_matrix;
    view_matrix.m[0][0] = cam->right.x;
    view_matrix.m[0][1] = cam->right.y;
    view_matrix.m[0][2] = cam->right.z;
    view_matrix.m[1][0] = cam->up.x;
    view_matrix.m[1][1] = cam->up.y;
    view_matrix.m[1][2] = cam->up.z;
    view_matrix.m[2][0] = cam->forward.x;
    view_matrix.m[2][1] = cam->forward.y;
    view_matrix.m[2][2] = cam->forward.z;
    
    mat3 cov_cam = mat3_mul(mat3_mul(view_matrix, g->covariance), mat3_transpose(view_matrix));
    
    // Jacobian of perspective projection
    float J[2][3];
    J[0][0] = focal / z;
    J[0][1] = 0;
    J[0][2] = -focal * x / (z * z);
    J[1][0] = 0;
    J[1][1] = -focal / z;
    J[1][2] = focal * y / (z * z);
    
    // Project covariance: cov2d = J * cov_cam * J^T
    float cov2d[2][2] = {0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    cov2d[i][j] += J[i][k] * cov_cam.m[k][l] * J[j][l];
                }
            }
        }
    }
    
    // Add small value for numerical stability
    cov2d[0][0] += 0.1f;
    cov2d[1][1] += 0.1f;
    
    g2d.cov[0] = cov2d[0][0];
    g2d.cov[1] = cov2d[0][1];
    g2d.cov[2] = cov2d[1][1];
    
    g2d.color[0] = g->color[0];
    g2d.color[1] = g->color[1];
    g2d.color[2] = g->color[2];
    g2d.opacity = g->opacity;
    g2d.valid = 1;
    
    return g2d;
}

// Comparison function for depth sorting (front to back)
int compare_depth(const void *a, const void *b) {
    const Gaussian2D *ga = (const Gaussian2D *)a;
    const Gaussian2D *gb = (const Gaussian2D *)b;
    
    if (!ga->valid) return 1;
    if (!gb->valid) return -1;
    
    if (ga->depth < gb->depth) return -1;
    if (ga->depth > gb->depth) return 1;
    return 0;
}

// Render scene from camera viewpoint
void render(Gaussian *gaussians, int count, Camera *cam, uint8_t *image) {
    // Project all Gaussians to 2D
    Gaussian2D *g2d = malloc(count * sizeof(Gaussian2D));
    
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        g2d[i] = project_gaussian(&gaussians[i], cam);
    }
    
    // Sort front to back for early ray termination
    qsort(g2d, count, sizeof(Gaussian2D), compare_depth);
    
    // Clear image to black
    memset(image, 0, WIDTH * HEIGHT * 3);
    
    // Render each pixel
    #pragma omp parallel for collapse(2)
    for (int py = 0; py < HEIGHT; py++) {
        for (int px = 0; px < WIDTH; px++) {
            float color[3] = {0, 0, 0};
            float T = 1.0f;  // Transmittance
            
            // Accumulate color from each Gaussian (front to back)
            for (int i = 0; i < count; i++) {
                if (!g2d[i].valid) continue;
                if (T < 0.01f) break;  // Early ray termination
                
                float dx = px - g2d[i].x;
                float dy = py - g2d[i].y;
                
                // Compute inverse of 2x2 covariance matrix
                float a = g2d[i].cov[0];
                float b = g2d[i].cov[1];
                float c = g2d[i].cov[2];
                
                float det = a * c - b * b;
                if (det < 1e-6f) continue;
                
                float inv_det = 1.0f / det;
                float inv_a = c * inv_det;
                float inv_b = -b * inv_det;
                float inv_c = a * inv_det;
                
                // Mahalanobis distance
                float dist = dx * (inv_a * dx + inv_b * dy) + dy * (inv_b * dx + inv_c * dy);
                
                // Gaussian weight
                float weight = expf(-0.5f * dist);
                float alpha = g2d[i].opacity * weight;
                
                // Alpha blending with transmittance
                color[0] += g2d[i].color[0] * alpha * T;
                color[1] += g2d[i].color[1] * alpha * T;
                color[2] += g2d[i].color[2] * alpha * T;
                T *= (1.0f - alpha);
            }
            
            // Write to image buffer
            int idx = (py * WIDTH + px) * 3;
            image[idx + 0] = (uint8_t)(fminf(color[0] * 255.0f, 255.0f));
            image[idx + 1] = (uint8_t)(fminf(color[1] * 255.0f, 255.0f));
            image[idx + 2] = (uint8_t)(fminf(color[2] * 255.0f, 255.0f));
        }
    }
    
    free(g2d);
}

// Save RGB image as WebP
void save_webp(const char *filename, uint8_t *image) {
    uint8_t *output;
    size_t size = WebPEncodeRGB(image, WIDTH, HEIGHT, WIDTH * 3, 75, &output);
    
    if (size == 0) {
        fprintf(stderr, "Error encoding WebP\n");
        return;
    }
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        WebPFree(output);
        return;
    }
    
    fwrite(output, 1, size, f);
    fclose(f);
    WebPFree(output);
    
    printf("Saved %s\n", filename);
}

int main() {
    srand(42);  // Fixed seed for reproducibility
    
    printf("Initializing %d Gaussians...\n", NUM_GAUSSIANS);
    Gaussian *gaussians = malloc(NUM_GAUSSIANS * sizeof(Gaussian));
    init_gaussians(gaussians, NUM_GAUSSIANS);
    
    uint8_t *image = malloc(WIDTH * HEIGHT * 3);
    
    // Render 4 views from random camera positions
    for (int view = 0; view < 4; view++) {
        printf("Rendering view %d...\n", view + 1);
        
        Camera cam = create_random_camera();
        render(gaussians, NUM_GAUSSIANS, &cam, image);
        
        char filename[64];
        sprintf(filename, "view_%d_splat.webp", view + 1);
        save_webp(filename, image);
    }
    
    free(image);
    free(gaussians);
    
    printf("Done!\n");
    return 0;
}