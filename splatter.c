#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
#include <webp/decode.h>
#include <webp/encode.h>
#include <webp/mux.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// MATH STRUCTURES
// ============================================================================

typedef struct { float x, y, z; } Vec3;
typedef struct { float u, v; } Vec2;
typedef struct { float m[4][4]; } Mat4;

// ============================================================================
// VEC3 OPERATIONS
// ============================================================================

Vec3 vec3_add(Vec3 a, Vec3 b) { 
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; 
}

Vec3 vec3_sub(Vec3 a, Vec3 b) { 
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; 
}

Vec3 vec3_mul(Vec3 a, float t) { 
    return (Vec3){a.x * t, a.y * t, a.z * t}; 
}

float vec3_dot(Vec3 a, Vec3 b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float vec3_length(Vec3 v) { 
    return sqrtf(vec3_dot(v, v)); 
}

Vec3 vec3_normalize(Vec3 v) { 
    float len = vec3_length(v);
    return (Vec3){v.x / len, v.y / len, v.z / len};
}

// ============================================================================
// MAT4 OPERATIONS
// ============================================================================

Mat4 mat4_identity(void) {
    Mat4 m = {0};
    m.m[0][0] = m.m[1][1] = m.m[2][2] = m.m[3][3] = 1.0f;
    return m;
}

Mat4 mat4_multiply(Mat4 a, Mat4 b) {
    Mat4 result = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return result;
}

Mat4 mat4_translation(Vec3 t) {
    Mat4 m = mat4_identity();
    m.m[0][3] = t.x;
    m.m[1][3] = t.y;
    m.m[2][3] = t.z;
    return m;
}

Mat4 mat4_rotation_x(float angle) {
    Mat4 m = mat4_identity();
    float c = cosf(angle);
    float s = sinf(angle);
    m.m[1][1] = c; m.m[1][2] = -s;
    m.m[2][1] = s; m.m[2][2] = c;
    return m;
}

Mat4 mat4_rotation_y(float angle) {
    Mat4 m = mat4_identity();
    float c = cosf(angle);
    float s = sinf(angle);
    m.m[0][0] = c;  m.m[0][2] = s;
    m.m[2][0] = -s; m.m[2][2] = c;
    return m;
}

Mat4 mat4_rotation_z(float angle) {
    Mat4 m = mat4_identity();
    float c = cosf(angle);
    float s = sinf(angle);
    m.m[0][0] = c; m.m[0][1] = -s;
    m.m[1][0] = s; m.m[1][1] = c;
    return m;
}

Vec3 mat4_transform_point(Mat4 m, Vec3 p) {
    float w = m.m[3][0] * p.x + m.m[3][1] * p.y + m.m[3][2] * p.z + m.m[3][3];
    return (Vec3){
        (m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3]) / w,
        (m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3]) / w,
        (m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3]) / w
    };
}

// ============================================================================
// GAUSSIAN STRUCTURES
// ============================================================================

typedef struct {
    Vec3 position;
    float radius;
    Vec3 color;
    float opacity;
} Gaussian;

typedef struct {
    Gaussian* gaussians;
    size_t count;
    size_t capacity;
    Vec3 position;
    Vec3 rotation;
} GaussianCloud;

typedef struct {
    Vec3 v0, v1, v2;
    Vec2 t0, t1, t2;
} Triangle;

typedef struct {
    Triangle* triangles;
    size_t triangle_count;
    unsigned char* texture_data;
    int texture_width;
    int texture_height;
} Mesh;

typedef struct {
    Vec3 position;
    Vec3 look_at;
    Vec3 up;
    float fov;
} Camera;

// ============================================================================
// MESH LOADING
// ============================================================================

Vec3 sample_texture(const Mesh* mesh, float u, float v) {
    u = u - floorf(u);
    v = v - floorf(v);
    int x = (int)(u * (mesh->texture_width - 1));
    int y = (int)(v * (mesh->texture_height - 1));
    int idx = (y * mesh->texture_width + x) * 4;
    return (Vec3){
        mesh->texture_data[idx] / 255.0f,
        mesh->texture_data[idx + 1] / 255.0f,
        mesh->texture_data[idx + 2] / 255.0f
    };
}

Mesh load_mesh(const char* obj_file, const char* texture_file) {
    Mesh mesh = {0};
    
    // Load OBJ
    Vec3* vertices = malloc(1000000 * sizeof(Vec3));
    Vec2* texcoords = malloc(1000000 * sizeof(Vec2));
    int vertex_count = 0, texcoord_count = 0;
    mesh.triangles = malloc(1000000 * sizeof(Triangle));
    mesh.triangle_count = 0;

    FILE* file = fopen(obj_file, "r");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", obj_file);
        free(vertices);
        free(texcoords);
        return mesh;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            sscanf(line + 2, "%f %f %f", 
                &vertices[vertex_count].x,
                &vertices[vertex_count].y,
                &vertices[vertex_count].z);
            vertex_count++;
        } else if (line[0] == 'v' && line[1] == 't') {
            sscanf(line + 3, "%f %f",
                &texcoords[texcoord_count].u,
                &texcoords[texcoord_count].v);
            texcoord_count++;
        } else if (line[0] == 'f') {
            int v1, v2, v3, t1, t2, t3, n1, n2, n3;
            sscanf(line + 2, "%d/%d/%d %d/%d/%d %d/%d/%d",
                &v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3);
            mesh.triangles[mesh.triangle_count].v0 = vertices[v1-1];
            mesh.triangles[mesh.triangle_count].v1 = vertices[v2-1];
            mesh.triangles[mesh.triangle_count].v2 = vertices[v3-1];
            mesh.triangles[mesh.triangle_count].t0 = texcoords[t1-1];
            mesh.triangles[mesh.triangle_count].t1 = texcoords[t2-1];
            mesh.triangles[mesh.triangle_count].t2 = texcoords[t3-1];
            mesh.triangle_count++;
        }
    }
    fclose(file);
    free(vertices);
    free(texcoords);

    // Load texture
    FILE* tex_file = fopen(texture_file, "rb");
    if (!tex_file) {
        fprintf(stderr, "Failed to open %s\n", texture_file);
        return mesh;
    }

    fseek(tex_file, 0, SEEK_END);
    size_t file_size = ftell(tex_file);
    fseek(tex_file, 0, SEEK_SET);

    uint8_t* file_data = malloc(file_size);
    fread(file_data, 1, file_size, tex_file);
    fclose(tex_file);

    mesh.texture_data = WebPDecodeRGBA(file_data, file_size,
                                       &mesh.texture_width,
                                       &mesh.texture_height);
    free(file_data);

    printf("Loaded %s: %zu triangles\n", obj_file, mesh.triangle_count);
    return mesh;
}

void free_mesh(Mesh* mesh) {
    if (mesh->triangles) free(mesh->triangles);
    if (mesh->texture_data) WebPFree(mesh->texture_data);
}

// ============================================================================
// GAUSSIAN CLOUD OPERATIONS
// ============================================================================

GaussianCloud create_cloud(size_t capacity) {
    GaussianCloud cloud = {0};
    cloud.gaussians = malloc(capacity * sizeof(Gaussian));
    cloud.capacity = capacity;
    cloud.count = 0;
    return cloud;
}

void add_gaussian(GaussianCloud* cloud, Gaussian g) {
    if (cloud->count >= cloud->capacity) {
        cloud->capacity *= 2;
        cloud->gaussians = realloc(cloud->gaussians, cloud->capacity * sizeof(Gaussian));
    }
    cloud->gaussians[cloud->count++] = g;
}

GaussianCloud mesh_to_gaussians(const char* obj_file, const char* texture_file, int samples) {
    Mesh mesh = load_mesh(obj_file, texture_file);
    GaussianCloud cloud = create_cloud(mesh.triangle_count * samples);

    for (size_t i = 0; i < mesh.triangle_count; i++) {
        Triangle* tri = &mesh.triangles[i];
        
        Vec3 edge1 = vec3_sub(tri->v1, tri->v0);
        Vec3 edge2 = vec3_sub(tri->v2, tri->v0);
        float area = vec3_length(vec3_cross(edge1, edge2)) * 0.5f;
        float size = sqrtf(area / samples) * 0.35f;  // Reduced from 0.7

        for (int s = 0; s < samples; s++) {
            float r1 = (float)rand() / (float)RAND_MAX;
            float r2 = (float)rand() / (float)RAND_MAX;
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }
            float r3 = 1.0f - r1 - r2;

            Vec3 pos = vec3_add(vec3_add(
                vec3_mul(tri->v0, r3),
                vec3_mul(tri->v1, r1)),
                vec3_mul(tri->v2, r2));

            Vec2 uv;
            uv.u = r3 * tri->t0.u + r1 * tri->t1.u + r2 * tri->t2.u;
            uv.v = r3 * tri->t0.v + r1 * tri->t1.v + r2 * tri->t2.v;

            Gaussian g;
            g.position = pos;
            g.radius = size;
            g.color = sample_texture(&mesh, uv.u, uv.v);
            g.opacity = 0.7f;  // Reduced from 0.9

            add_gaussian(&cloud, g);
        }
    }

    free_mesh(&mesh);
    printf("Created cloud: %zu gaussians\n", cloud.count);
    return cloud;
}

void free_cloud(GaussianCloud* cloud) {
    if (cloud->gaussians) free(cloud->gaussians);
    cloud->gaussians = NULL;
}

// ============================================================================
// RENDERING
// ============================================================================

typedef struct {
    float depth;
    int index;
} DepthIndex;

int compare_depth(const void* a, const void* b) {
    // Sort BACK TO FRONT (larger depth first)
    float diff = ((DepthIndex*)a)->depth - ((DepthIndex*)b)->depth;
    return (diff < 0) ? 1 : (diff > 0 ? -1 : 0);
}

void render_frame(GaussianCloud* clouds, int cloud_count, Camera* camera,
                  unsigned char* frame, int width, int height) {
    float aspect = (float)width / height;
    
    // Camera basis vectors - MATCH RAYTRACER
    Vec3 forward = vec3_normalize(vec3_sub(camera->look_at, camera->position));
    Vec3 right = vec3_normalize(vec3_cross(forward, camera->up));
    Vec3 up = vec3_cross(right, forward);
    float scale = tanf((camera->fov * 0.5f) * M_PI / 180.0f);

    // Count total gaussians
    size_t total = 0;
    for (int c = 0; c < cloud_count; c++) {
        total += clouds[c].count;
    }

    // Project all gaussians
    DepthIndex* indices = malloc(total * sizeof(DepthIndex));
    float* screen_x = malloc(total * sizeof(float));
    float* screen_y = malloc(total * sizeof(float));
    float* screen_radius = malloc(total * sizeof(float));
    Vec3* colors = malloc(total * sizeof(Vec3));
    float* opacities = malloc(total * sizeof(float));

    size_t idx = 0;
    for (int c = 0; c < cloud_count; c++) {
        Mat4 rot_x = mat4_rotation_x(clouds[c].rotation.x);
        Mat4 rot_y = mat4_rotation_y(clouds[c].rotation.y);
        Mat4 rot_z = mat4_rotation_z(clouds[c].rotation.z);
        Mat4 trans = mat4_translation(clouds[c].position);
        Mat4 transform = mat4_multiply(trans, mat4_multiply(rot_z,
                        mat4_multiply(rot_y, rot_x)));

        for (size_t i = 0; i < clouds[c].count; i++) {
            Gaussian* g = &clouds[c].gaussians[i];
            Vec3 world_pos = mat4_transform_point(transform, g->position);

            // Transform to camera space
            Vec3 cam_pos = vec3_sub(world_pos, camera->position);
            float x = vec3_dot(cam_pos, right);
            float y = vec3_dot(cam_pos, up);
            float z = vec3_dot(cam_pos, forward);

            if (z > 0.1f) {
                indices[idx].depth = z;
                indices[idx].index = idx;
                
                // Project to NDC then to screen - MATCH RAYTRACER
                float ndc_x = x / (z * aspect * scale);
                float ndc_y = -y / (z * scale);  // Note the negation for Y
                
                screen_x[idx] = (ndc_x + 1.0f) * 0.5f;
                screen_y[idx] = (1.0f - ndc_y) * 0.5f;  // Flip Y for screen coords
                
                // Perspective-correct radius
                screen_radius[idx] = g->radius / (z * scale);
                
                colors[idx] = g->color;
                opacities[idx] = g->opacity;
                idx++;
            }
        }
    }
    total = idx;

    // Sort by depth (back to front)
    qsort(indices, total, sizeof(DepthIndex), compare_depth);

    // Clear frame
    memset(frame, 50, width * height * 3);

    // Render gaussians with proper alpha blending
    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float px = (float)x / width;
            float py = (float)y / height;

            Vec3 color = {0.05f, 0.05f, 0.05f};  // Background
            float transmittance = 1.0f;

            // Blend gaussians back to front
            for (size_t i = 0; i < total; i++) {
                if (transmittance < 0.01f) break;  // Early termination

                int idx = indices[i].index;
                
                float dx = px - screen_x[idx];
                float dy = py - screen_y[idx];
                float dist2 = (dx * dx + dy * dy) / (screen_radius[idx] * screen_radius[idx]);

                if (dist2 < 9.0f) {  // 3 sigma cutoff
                    float gauss = expf(-0.5f * dist2);
                    float alpha = opacities[idx] * gauss;
                    
                    // Alpha blend
                    color = vec3_add(color, vec3_mul(colors[idx], alpha * transmittance));
                    transmittance *= (1.0f - alpha);
                }
            }

            int pidx = (y * width + x) * 3;
            frame[pidx] = (unsigned char)fminf(color.x * 255.0f, 255.0f);
            frame[pidx + 1] = (unsigned char)fminf(color.y * 255.0f, 255.0f);
            frame[pidx + 2] = (unsigned char)fminf(color.z * 255.0f, 255.0f);
        }
    }

    free(indices);
    free(screen_x);
    free(screen_y);
    free(screen_radius);
    free(colors);
    free(opacities);
}

// ============================================================================
// PROGRESS BAR
// ============================================================================

void update_progress(int frame, int total, clock_t start) {
    printf("\r[");
    int bar_width = 30;
    int pos = bar_width * (frame + 1) / total;
    
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }

    float progress = (frame + 1.0f) / total * 100.0f;
    float elapsed = (clock() - start) / (float)CLOCKS_PER_SEC;
    float eta = elapsed * total / (frame + 1) - elapsed;

    printf("] %.1f%% | %d/%d | %.1fs | ETA %.1fs", 
        progress, frame + 1, total, elapsed, eta);
    fflush(stdout);

    if (frame == total - 1) printf("\n");
}

// ============================================================================
// IMAGE SCALING
// ============================================================================

float cubic_hermite(float A, float B, float C, float D, float t) {
    float a = -A/2.0f + (3.0f*B)/2.0f - (3.0f*C)/2.0f + D/2.0f;
    float b = A - (5.0f*B)/2.0f + 2.0f*C - D/2.0f;
    float c = -A/2.0f + C/2.0f;
    float d = B;
    return a*t*t*t + b*t*t + c*t + d;
}

uint32_t get_pixel(unsigned char* frame, int x, int y, int w, int h) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= w) x = w - 1;
    if (y >= h) y = h - 1;
    int idx = (y * w + x) * 3;
    return (frame[idx] << 16) | (frame[idx + 1] << 8) | frame[idx + 2];
}

uint32_t bicubic(unsigned char* frame, float x, float y, int w, int h) {
    int x1 = (int)x;
    int y1 = (int)y;
    float fx = x - x1;
    float fy = y - y1;

    uint32_t p[4][4];
    for (int dy = -1; dy <= 2; dy++) {
        for (int dx = -1; dx <= 2; dx++) {
            p[dy+1][dx+1] = get_pixel(frame, x1 + dx, y1 + dy, w, h);
        }
    }

    float r[4], g[4], b[4];
    for (int i = 0; i < 4; i++) {
        r[i] = cubic_hermite((p[i][0]>>16)&0xFF, (p[i][1]>>16)&0xFF,
                             (p[i][2]>>16)&0xFF, (p[i][3]>>16)&0xFF, fx);
        g[i] = cubic_hermite((p[i][0]>>8)&0xFF, (p[i][1]>>8)&0xFF,
                             (p[i][2]>>8)&0xFF, (p[i][3]>>8)&0xFF, fx);
        b[i] = cubic_hermite(p[i][0]&0xFF, p[i][1]&0xFF,
                             p[i][2]&0xFF, p[i][3]&0xFF, fx);
    }

    int ri = (int)(cubic_hermite(r[0], r[1], r[2], r[3], fy) + 0.5f);
    int gi = (int)(cubic_hermite(g[0], g[1], g[2], g[3], fy) + 0.5f);
    int bi = (int)(cubic_hermite(b[0], b[1], b[2], b[3], fy) + 0.5f);

    ri = ri < 0 ? 0 : (ri > 255 ? 255 : ri);
    gi = gi < 0 ? 0 : (gi > 255 ? 255 : gi);
    bi = bi < 0 ? 0 : (bi > 255 ? 255 : bi);

    return (0xFF << 24) | (ri << 16) | (gi << 8) | bi;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    srand(time(NULL));
    
    int width = 800, height = 600;
    int render_width = (int)(width * 0.9f);
    int render_height = (int)(height * 0.9f);
    int fps = 24;
    int duration_ms = 4000;
    int frame_count = (duration_ms * fps) / 1000;

    Camera camera = {
        .position = {-3.0f, 3.0f, -3.0f},
        .look_at = {0.0f, 0.0f, 0.0f},
        .up = {0.0f, 1.0f, 0.0f},
        .fov = 60.0f
    };

    // Create clouds
    printf("Loading meshes and creating gaussian clouds...\n");
    GaussianCloud clouds[3];
    clouds[0] = mesh_to_gaussians("drone.obj", "drone.webp", 5);  // Increased samples
    clouds[1] = mesh_to_gaussians("treasure.obj", "treasure.webp", 4);
    clouds[2] = mesh_to_gaussians("ground.obj", "ground.webp", 3);

    // Allocate frames
    unsigned char** frames = malloc(frame_count * sizeof(unsigned char*));
    for (int i = 0; i < frame_count; i++) {
        frames[i] = malloc(width * height * 3);
    }

    // Render frames
    printf("Rendering %d frames...\n", frame_count);
    clock_t start = clock();

    for (int f = 0; f < frame_count; f++) {
        float t = f * (2.0f * M_PI / 120.0f);

        clouds[0].position = (Vec3){2.0f * cosf(t), 1.0f + 0.2f * sinf(2*t), 2.0f * sinf(t)};
        clouds[0].rotation = (Vec3){0.1f * sinf(t), t, 0.1f * cosf(t)};
        
        clouds[1].position = (Vec3){1.0f, 0.5f + 0.1f * sinf(t), 1.0f};
        clouds[1].rotation = (Vec3){0, t * 0.5f, 0};

        unsigned char* render_frame_buf = malloc(render_width * render_height * 3);
        render_frame(clouds, 3, &camera, render_frame_buf, render_width, render_height);

        // Scale up
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sx = x * (render_width - 1.0f) / (width - 1.0f);
                float sy = y * (render_height - 1.0f) / (height - 1.0f);
                uint32_t pixel = bicubic(render_frame_buf, sx, sy, render_width, render_height);
                
                int idx = (y * width + x) * 3;
                frames[f][idx] = (pixel >> 16) & 0xFF;
                frames[f][idx + 1] = (pixel >> 8) & 0xFF;
                frames[f][idx + 2] = pixel & 0xFF;
            }
        }
        free(render_frame_buf);

        update_progress(f, frame_count, start);
    }

    // Save animation
    printf("Encoding WebP animation...\n");
    WebPAnimEncoderOptions anim_config;
    WebPAnimEncoderOptionsInit(&anim_config);
    WebPAnimEncoder* enc = WebPAnimEncoderNew(width, height, &anim_config);

    WebPConfig config;
    WebPConfigInit(&config);
    config.quality = 90;

    WebPPicture pic;
    WebPPictureInit(&pic);
    pic.width = width;
    pic.height = height;
    pic.use_argb = 1;
    WebPPictureAlloc(&pic);

    for (int f = 0; f < frame_count; f++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                pic.argb[y * width + x] = (0xFF << 24) |
                    (frames[f][idx] << 16) |
                    (frames[f][idx + 1] << 8) |
                    frames[f][idx + 2];
            }
        }
        int timestamp = f * (duration_ms / frame_count);
        WebPAnimEncoderAdd(enc, &pic, timestamp, &config);
    }

    WebPAnimEncoderAdd(enc, NULL, duration_ms, NULL);
    WebPData webp_data;
    WebPDataInit(&webp_data);
    WebPAnimEncoderAssemble(enc, &webp_data);

    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_splat.webp", localtime(&now));
    
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(webp_data.bytes, webp_data.size, 1, fp);
        fclose(fp);
        printf("Saved: %s\n", filename);
    }

    // Cleanup
    WebPDataClear(&webp_data);
    WebPAnimEncoderDelete(enc);
    WebPPictureFree(&pic);

    for (int i = 0; i < frame_count; i++) {
        free(frames[i]);
    }
    free(frames);

    for (int i = 0; i < 3; i++) {
        free_cloud(&clouds[i]);
    }

    return 0;
}