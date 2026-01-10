#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <webp/encode.h>

#define WIDTH 800
#define HEIGHT 600
#define NUM_GAUSSIANS 1000

// === DATA STRUCTURES ===

typedef struct {
    float x, y, z;
} vec3;

typedef struct {
    float m[3][3];
} mat3;

// 3D Gaussian: g_i = (μ_i, Σ_i, α_i, c_i)
typedef struct {
    vec3 mu;           // μ_i ∈ ℝ³ - position
    mat3 Sigma;        // Σ_i ∈ ℝ³ˣ³ - covariance (SPD)
    float alpha;       // α_i ∈ (0,1] - opacity
    float c[3];        // c_i ∈ [0,1]³ - RGB color
} Gaussian;

// Camera parameters
typedef struct {
    vec3 t_cam;        // Camera position
    mat3 R_cam;        // Camera rotation (world to camera)
    float fx, fy;      // Focal lengths
    float cx, cy;      // Principal point
} Camera;

// Projected 2D Gaussian
typedef struct {
    float mu_2d[2];    // μ^2D - 2D mean
    float Sigma_2d[3]; // Σ^2D stored as [σ₁₁, σ₁₂, σ₂₂]
    float c[3];        // Color
    float alpha;       // Opacity
    float depth;       // z_i for sorting
    int valid;         // Culling flag
} Gaussian2D;

// === UTILITY FUNCTIONS ===

float randf(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

vec3 vec3_add(vec3 a, vec3 b) {
    return (vec3){a.x + b.x, a.y + b.y, a.z + b.z};
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

// Matrix multiplication: C = A * B
mat3 mat3_mul(mat3 A, mat3 B) {
    mat3 C = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                C.m[i][j] += A.m[i][k] * B.m[k][j];
            }
        }
    }
    return C;
}

// Matrix transpose
mat3 mat3_transpose(mat3 M) {
    mat3 Mt;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Mt.m[i][j] = M.m[j][i];
        }
    }
    return Mt;
}

// Transform vector by rotation matrix: v' = R * v
vec3 mat3_vec3_mul(mat3 R, vec3 v) {
    return (vec3){
        R.m[0][0] * v.x + R.m[0][1] * v.y + R.m[0][2] * v.z,
        R.m[1][0] * v.x + R.m[1][1] * v.y + R.m[1][2] * v.z,
        R.m[2][0] * v.x + R.m[2][1] * v.y + R.m[2][2] * v.z
    };
}

// === INITIALIZATION ===

// Create covariance: Σ = R * S * S^T * R^T
mat3 create_covariance(float scale_x, float scale_y, float scale_z, float rx, float ry, float rz) {
    // Rotation matrix from Euler angles
    float cx = cosf(rx), sx = sinf(rx);
    float cy = cosf(ry), sy = sinf(ry);
    float cz = cosf(rz), sz = sinf(rz);
    
    mat3 R;
    R.m[0][0] = cy * cz;
    R.m[0][1] = -cy * sz;
    R.m[0][2] = sy;
    R.m[1][0] = sx * sy * cz + cx * sz;
    R.m[1][1] = -sx * sy * sz + cx * cz;
    R.m[1][2] = -sx * cy;
    R.m[2][0] = -cx * sy * cz + sx * sz;
    R.m[2][1] = cx * sy * sz + sx * cz;
    R.m[2][2] = cx * cy;
    
    // Scale matrix S
    mat3 S = {0};
    S.m[0][0] = scale_x;
    S.m[1][1] = scale_y;
    S.m[2][2] = scale_z;
    
    // Σ = R * S * S^T * R^T
    mat3 RS = mat3_mul(R, S);
    mat3 St = mat3_transpose(S);
    mat3 RSSt = mat3_mul(RS, St);
    return mat3_mul(RSSt, mat3_transpose(R));
}

void init_gaussians(Gaussian *gaussians, int count) {
    for (int i = 0; i < count; i++) {
        gaussians[i].mu = (vec3){
            randf(-5.0f, 5.0f),
            randf(-5.0f, 5.0f),
            randf(-5.0f, 5.0f)
        };
        
        float scale_x = randf(0.05f, 0.3f);
        float scale_y = randf(0.05f, 0.3f);
        float scale_z = randf(0.05f, 0.3f);
        float rot_x = randf(0, 2 * M_PI);
        float rot_y = randf(0, 2 * M_PI);
        float rot_z = randf(0, 2 * M_PI);
        gaussians[i].Sigma = create_covariance(scale_x, scale_y, scale_z, rot_x, rot_y, rot_z);
        
        gaussians[i].c[0] = randf(0.0f, 1.0f);
        gaussians[i].c[1] = randf(0.0f, 1.0f);
        gaussians[i].c[2] = randf(0.0f, 1.0f);
        
        gaussians[i].alpha = randf(0.3f, 0.9f);
    }
}

// === CAMERA SETUP ===

Camera create_camera(vec3 position, vec3 look_at) {
    Camera cam;
    
    // Camera position
    cam.t_cam = position;
    
    // Build camera frame
    vec3 forward = vec3_normalize(vec3_sub(look_at, position));
    vec3 world_up = {0, 1, 0};
    vec3 right = vec3_normalize(vec3_cross(forward, world_up));
    vec3 up = vec3_cross(right, forward);
    
    // R_cam = [right; up; forward] (rows are camera axes)
    cam.R_cam.m[0][0] = right.x;
    cam.R_cam.m[0][1] = right.y;
    cam.R_cam.m[0][2] = right.z;
    cam.R_cam.m[1][0] = up.x;
    cam.R_cam.m[1][1] = up.y;
    cam.R_cam.m[1][2] = up.z;
    cam.R_cam.m[2][0] = forward.x;
    cam.R_cam.m[2][1] = forward.y;
    cam.R_cam.m[2][2] = forward.z;
    
    // Intrinsics
    float fov = 60.0f * M_PI / 180.0f;
    cam.fx = cam.fy = HEIGHT / (2.0f * tanf(fov / 2.0f));
    cam.cx = WIDTH / 2.0f;
    cam.cy = HEIGHT / 2.0f;
    
    return cam;
}

Camera create_random_camera() {
    float theta = randf(0, 2 * M_PI);
    float phi = randf(-M_PI / 6, M_PI / 6);
    float radius = randf(8.0f, 12.0f);
    
    vec3 position = {
        radius * cosf(phi) * cosf(theta),
        radius * sinf(phi),
        radius * cosf(phi) * sinf(theta)
    };
    
    vec3 look_at = {0, 0, 0};
    
    return create_camera(position, look_at);
}

// === PROJECTION (Following formalization exactly) ===

Gaussian2D project_gaussian(Gaussian *g, Camera *cam) {
    Gaussian2D g2d = {0};
    
    // Step 1: Transform mean to camera space
    // μ^cam = R_cam * (μ^world - t_cam)
    vec3 mu_world_shifted = vec3_sub(g->mu, cam->t_cam);
    vec3 mu_cam = mat3_vec3_mul(cam->R_cam, mu_world_shifted);
    
    float x = mu_cam.x;
    float y = mu_cam.y;
    float z = mu_cam.z;
    
    // Cull if behind camera
    if (z < 0.1f) {
        g2d.valid = 0;
        return g2d;
    }
    
    g2d.depth = z;
    
    // Step 2: Transform covariance to camera space
    // Σ^cam = R_cam * Σ^world * R_cam^T
    mat3 R_cam_T = mat3_transpose(cam->R_cam);
    mat3 temp = mat3_mul(cam->R_cam, g->Sigma);
    mat3 Sigma_cam = mat3_mul(temp, R_cam_T);
    
    // Step 3: Project mean to 2D
    // μ^2D = [fx * x/z + cx, fy * y/z + cy]
    g2d.mu_2d[0] = cam->fx * x / z + cam->cx;
    g2d.mu_2d[1] = cam->fy * y / z + cam->cy;
    
    // Step 4: Compute Jacobian
    // J = [[fx/z, 0, -fx*x/z²],
    //      [0, fy/z, -fy*y/z²]]
    float J[2][3];
    J[0][0] = cam->fx / z;
    J[0][1] = 0.0f;
    J[0][2] = -cam->fx * x / (z * z);
    J[1][0] = 0.0f;
    J[1][1] = cam->fy / z;
    J[1][2] = -cam->fy * y / (z * z);
    
    // Step 5: Project covariance to 2D
    // Σ^2D = J * Σ^cam * J^T
    float Sigma_2d[2][2] = {0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    Sigma_2d[i][j] += J[i][k] * Sigma_cam.m[k][l] * J[j][l];
                }
            }
        }
    }
    
    // Add small regularization for numerical stability
    Sigma_2d[0][0] += 0.1f;
    Sigma_2d[1][1] += 0.1f;
    
    // Store as [σ₁₁, σ₁₂, σ₂₂]
    g2d.Sigma_2d[0] = Sigma_2d[0][0];
    g2d.Sigma_2d[1] = Sigma_2d[0][1];
    g2d.Sigma_2d[2] = Sigma_2d[1][1];
    
    g2d.c[0] = g->c[0];
    g2d.c[1] = g->c[1];
    g2d.c[2] = g->c[2];
    g2d.alpha = g->alpha;
    g2d.valid = 1;
    
    return g2d;
}

// === SORTING ===

int compare_depth(const void *a, const void *b) {
    const Gaussian2D *ga = (const Gaussian2D *)a;
    const Gaussian2D *gb = (const Gaussian2D *)b;
    
    if (!ga->valid) return 1;
    if (!gb->valid) return -1;
    
    // Front to back: smaller depth first
    if (ga->depth < gb->depth) return -1;
    if (ga->depth > gb->depth) return 1;
    return 0;
}

// === RENDERING (Alpha Compositing) ===

void render(Gaussian *gaussians, int count, Camera *cam, uint8_t *image) {
    // Project all Gaussians to 2D
    Gaussian2D *g2d = malloc(count * sizeof(Gaussian2D));
    
    for (int i = 0; i < count; i++) {
        g2d[i] = project_gaussian(&gaussians[i], cam);
    }
    
    // Sort by depth (front to back)
    qsort(g2d, count, sizeof(Gaussian2D), compare_depth);
    
    // Clear image
    memset(image, 0, WIDTH * HEIGHT * 3);
    
    // Render each pixel
    #pragma omp parallel for collapse(2)
    for (int v = 0; v < HEIGHT; v++) {
        for (int u = 0; u < WIDTH; u++) {
            float C[3] = {0, 0, 0};  // Accumulated color
            float T = 1.0f;           // Transmittance
            
            // Alpha composite in front-to-back order
            for (int i = 0; i < count; i++) {
                if (!g2d[i].valid) continue;
                if (T < 1e-4f) break;  // Early stopping
                
                // Compute offset: δ = [u - μ^2D[0], v - μ^2D[1]]
                float delta[2];
                delta[0] = (float)u - g2d[i].mu_2d[0];
                delta[1] = (float)v - g2d[i].mu_2d[1];
                
                // Invert 2x2 covariance matrix
                // Σ^2D = [[a, b], [b, c]]
                float a = g2d[i].Sigma_2d[0];
                float b = g2d[i].Sigma_2d[1];
                float c = g2d[i].Sigma_2d[2];
                
                float det = a * c - b * b;
                if (det < 1e-6f) continue;
                
                float inv_det = 1.0f / det;
                float Sigma_inv[2][2];
                Sigma_inv[0][0] = c * inv_det;
                Sigma_inv[0][1] = -b * inv_det;
                Sigma_inv[1][0] = -b * inv_det;
                Sigma_inv[1][1] = a * inv_det;
                
                // Mahalanobis distance: d² = δ^T * Σ^-1 * δ
                float d_sq = delta[0] * (Sigma_inv[0][0] * delta[0] + Sigma_inv[0][1] * delta[1])
                           + delta[1] * (Sigma_inv[1][0] * delta[0] + Sigma_inv[1][1] * delta[1]);
                
                // Evaluate 2D Gaussian: G^2D = exp(-d²/2)
                float G_2d = expf(-0.5f * d_sq);
                
                // Weight: w = α * G^2D
                float w = g2d[i].alpha * G_2d;
                
                // Alpha compositing:
                // C = C + T * w * c
                // T = T * (1 - w)
                C[0] += T * w * g2d[i].c[0];
                C[1] += T * w * g2d[i].c[1];
                C[2] += T * w * g2d[i].c[2];
                T *= (1.0f - w);
            }
            
            // Write to image buffer (clamp to [0, 255])
            int idx = (v * WIDTH + u) * 3;
            image[idx + 0] = (uint8_t)(fminf(C[0] * 255.0f, 255.0f));
            image[idx + 1] = (uint8_t)(fminf(C[1] * 255.0f, 255.0f));
            image[idx + 2] = (uint8_t)(fminf(C[2] * 255.0f, 255.0f));
        }
    }
    
    free(g2d);
}

// === FILE I/O ===

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

// === MAIN ===

int main() {
    srand(42);
    
    printf("Initializing %d Gaussians...\n", NUM_GAUSSIANS);
    Gaussian *gaussians = malloc(NUM_GAUSSIANS * sizeof(Gaussian));
    init_gaussians(gaussians, NUM_GAUSSIANS);
    
    uint8_t *image = malloc(WIDTH * HEIGHT * 3);
    
    // Render 4 views
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