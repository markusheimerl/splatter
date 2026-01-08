// splatter.c
//
// Multi-view patch-consistent Gaussian initialization + splat renderer.
// This revision focuses on fixing the remaining “point cloud” look by:
//
//  1) Making splats fatter/overlapping in a *controlled* way:
//     - Increase TARGET_SCREEN_SIGMA_PX (world scale chosen so projected sigma ~ this).
//     - Add a small extra blur term in screen space at render time (RENDER_ADDITIONAL_SIGMA_PX).
//  2) Slightly stronger opacity for well-supported points.
//  3) Render with a wider bbox and include more Gaussian tails.
//
// Still not full 3DGS training, but this should look much more like a continuous surface.
//
// Build (Makefile should link data.c):
//   clang -O3 -march=native -Wall -Wextra -fopenmp splatter.c data.c -lm -fopenmp -lpng -ljson-c -o splatter.out

#include "data.h"
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <float.h>

#define MAX_GAUSSIANS 400000
#define SH_C0 0.28209479177387814f

// ---- Initialization settings ----
#define REF_CAM_COUNT          12
#define NEIGHBOR_CAM_COUNT     20
#define INIT_SAMPLES_PER_REF   2600
#define MIN_VIEWS_CONSISTENT    5

// Inverse-depth search range (tune if needed)
#define INVDEPTH_NEAR  (1.0f/6.0f)  // far = 6
#define INVDEPTH_FAR   (1.0f/1.0f)  // near = 1

#define DEPTH_COARSE_SAMPLES  28
#define DEPTH_REFINE_SAMPLES  12
#define REFINE_WINDOW_FRAC    0.15f

// Photometric consistency
#define PATCH_RADIUS           1    // 3x3
#define COLOR_DIST_THRESH     0.18f
#define SCORE_SOFTEN          0.10f

// ---- Coverage / "point cloud" fixes ----
// Make splats larger in screen space (more overlap)
#define TARGET_SCREEN_SIGMA_PX     2.20f

// Add extra blur *in pixel units* at render time (fills holes without moving points)
// Covariance is in px^2, so we add blur^2 to diagonal.
#define RENDER_ADDITIONAL_SIGMA_PX 0.90f

// World-space scale clamps
#define SCALE_MIN_WORLD        0.0045f
#define SCALE_MAX_WORLD        0.1200f

// Rendering footprint
#define BBOX_SIGMAS            4.0f
#define POWER_CUTOFF          -9.0f  // include more tails than -6

typedef struct {
    float pos[3];
    float scale[3];
    float rot[4];
    float opacity;     // treated as logit in sigmoid()
    float sh[16][3];   // only sh[0] used
} Gaussian;

typedef struct {
    Gaussian *gaussians;
    int num_gaussians;
} GaussianModel;

typedef struct {
    int index;
    float depth;
    float mean2d[2];
    float cov2d[4];
    float color[3];
    float alpha;
} ProjectedGaussian;

static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static inline void quaternion_to_rotation_matrix(const float *q, float *R) {
    float w = q[0], x = q[1], y = q[2], z = q[3];

    R[0] = 1.0f - 2.0f*(y*y + z*z);
    R[1] = 2.0f*(x*y - w*z);
    R[2] = 2.0f*(x*z + w*y);

    R[3] = 2.0f*(x*y + w*z);
    R[4] = 1.0f - 2.0f*(x*x + z*z);
    R[5] = 2.0f*(y*z - w*x);

    R[6] = 2.0f*(x*z - w*y);
    R[7] = 2.0f*(y*z + w*x);
    R[8] = 1.0f - 2.0f*(x*x + y*y);
}

static inline void compute_covariance_3d(const Gaussian *g, float *cov3d) {
    float R[9];
    quaternion_to_rotation_matrix(g->rot, R);

    float S[9] = {
        g->scale[0], 0, 0,
        0, g->scale[1], 0,
        0, 0, g->scale[2]
    };

    float RS[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            RS[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) RS[i*3+j] += R[i*3+k] * S[k*3+j];
        }
    }

    // cov = RS * RS^T
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov3d[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) cov3d[i*3+j] += RS[i*3+k] * RS[j*3+k];
        }
    }
}

// cam->rotation here behaves as camera-to-world (consistent with generate_ray in data.c).
// So world->camera uses R^T.
static inline void project_point_to_camera(const float *point, const Camera *cam,
                                           float *px, float *py, float *depth_out) {
    float t[3];
    t[0] = point[0] - cam->position[0];
    t[1] = point[1] - cam->position[1];
    t[2] = point[2] - cam->position[2];

    float cam_space[3];
    cam_space[0] = cam->rotation[0]*t[0] + cam->rotation[3]*t[1] + cam->rotation[6]*t[2];
    cam_space[1] = cam->rotation[1]*t[0] + cam->rotation[4]*t[1] + cam->rotation[7]*t[2];
    cam_space[2] = cam->rotation[2]*t[0] + cam->rotation[5]*t[1] + cam->rotation[8]*t[2];

    float depth = -cam_space[2];
    *depth_out = depth;

    if (depth <= 0.0f) {
        *px = -1000.0f;
        *py = -1000.0f;
        return;
    }

    *px = cam->focal * cam_space[0] / (-cam_space[2]) + cam->width * 0.5f;
    *py = -cam->focal * cam_space[1] / (-cam_space[2]) + cam->height * 0.5f;
}

static inline void sample_bilinear_rgb(const Image *img, float x, float y, float *rgb) {
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;
    if (x > (float)(img->width - 1)) x = (float)(img->width - 1);
    if (y > (float)(img->height - 1)) y = (float)(img->height - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1; if (x1 >= img->width) x1 = img->width - 1;
    int y1 = y0 + 1; if (y1 >= img->height) y1 = img->height - 1;

    float tx = x - (float)x0;
    float ty = y - (float)y0;

    int i00 = (y0 * img->width + x0) * 4;
    int i10 = (y0 * img->width + x1) * 4;
    int i01 = (y1 * img->width + x0) * 4;
    int i11 = (y1 * img->width + x1) * 4;

    float r00 = img->data[i00+0] / 255.0f, g00 = img->data[i00+1] / 255.0f, b00 = img->data[i00+2] / 255.0f;
    float r10 = img->data[i10+0] / 255.0f, g10 = img->data[i10+1] / 255.0f, b10 = img->data[i10+2] / 255.0f;
    float r01 = img->data[i01+0] / 255.0f, g01 = img->data[i01+1] / 255.0f, b01 = img->data[i01+2] / 255.0f;
    float r11 = img->data[i11+0] / 255.0f, g11 = img->data[i11+1] / 255.0f, b11 = img->data[i11+2] / 255.0f;

    float r0 = r00*(1-tx) + r10*tx;
    float g0 = g00*(1-tx) + g10*tx;
    float b0 = b00*(1-tx) + b10*tx;

    float r1 = r01*(1-tx) + r11*tx;
    float g1 = g01*(1-tx) + g11*tx;
    float b1 = b01*(1-tx) + b11*tx;

    rgb[0] = r0*(1-ty) + r1*ty;
    rgb[1] = g0*(1-ty) + g1*ty;
    rgb[2] = b0*(1-ty) + b1*ty;
}

static inline float color_dist3(const float *a, const float *b) {
    float dr = a[0]-b[0], dg = a[1]-b[1], db = a[2]-b[2];
    return sqrtf(dr*dr + dg*dg + db*db);
}

static float patch_distance_3x3(const Image *img, float x, float y, const float *ref_rgb) {
    float acc = 0.0f;
    int n = 0;
    for (int oy = -PATCH_RADIUS; oy <= PATCH_RADIUS; oy++) {
        for (int ox = -PATCH_RADIUS; ox <= PATCH_RADIUS; ox++) {
            float rgb[3];
            sample_bilinear_rgb(img, x + (float)ox, y + (float)oy, rgb);
            acc += color_dist3(rgb, ref_rgb);
            n++;
        }
    }
    return acc / (float)n;
}

static inline float score_from_patchdist(float d) {
    float t = COLOR_DIST_THRESH;
    if (d >= t) return 0.0f;
    float x = (t - d) / (t + SCORE_SOFTEN);
    return clampf(x, 0.0f, 1.0f);
}

static void compute_nearest_cameras(const Dataset *ds, int ref_idx, int *out_ids, int k) {
    const float *p0 = ds->cameras[ref_idx]->position;

    typedef struct { int idx; float d2; } Item;
    Item *items = (Item*)malloc((size_t)ds->num_images * sizeof(Item));
    int n = 0;

    for (int i = 0; i < ds->num_images; i++) {
        if (i == ref_idx) continue;
        const float *p = ds->cameras[i]->position;
        float dx = p[0]-p0[0], dy = p[1]-p0[1], dz = p[2]-p0[2];
        items[n].idx = i;
        items[n].d2 = dx*dx + dy*dy + dz*dz;
        n++;
    }

    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i+1; j < n; j++) if (items[j].d2 < items[best].d2) best = j;
        Item tmp = items[i]; items[i] = items[best]; items[best] = tmp;
        out_ids[i] = items[i].idx;
    }

    free(items);
}

static bool find_best_depth_multiview(const Dataset *ds,
                                      int ref_cam_idx, int px, int py,
                                      const int *nbr_ids, int nbr_count,
                                      float *out_depth, float out_color_avg[3],
                                      int *out_support) {
    Image *ref_img = ds->images[ref_cam_idx];
    Camera *ref_cam = ds->cameras[ref_cam_idx];

    float ref_rgb[3];
    sample_bilinear_rgb(ref_img, (float)px, (float)py, ref_rgb);

    float ray_o[3], ray_d[3];
    generate_ray(ref_cam, px, py, ray_o, ray_d);

    float best_score = 0.0f;
    float best_invz = 0.0f;
    int best_support = 0;
    float best_col[3] = {ref_rgb[0], ref_rgb[1], ref_rgb[2]};

    // coarse inverse-depth
    for (int s = 0; s < DEPTH_COARSE_SAMPLES; s++) {
        float a = (float)s / (float)(DEPTH_COARSE_SAMPLES - 1);
        float invz = INVDEPTH_FAR + a * (INVDEPTH_NEAR - INVDEPTH_FAR);
        float depth = 1.0f / invz;

        float point[3] = {
            ray_o[0] + depth * ray_d[0],
            ray_o[1] + depth * ray_d[1],
            ray_o[2] + depth * ray_d[2]
        };

        int support = 0;
        float score = 0.0f;
        float col_sum[3] = {ref_rgb[0], ref_rgb[1], ref_rgb[2]};
        int col_n = 1;

        for (int ni = 0; ni < nbr_count; ni++) {
            int oi = nbr_ids[ni];
            float u, v, dcam;
            project_point_to_camera(point, ds->cameras[oi], &u, &v, &dcam);
            if (dcam <= 0.1f) continue;
            if (u < 1.0f || u > ds->images[oi]->width - 2.0f ||
                v < 1.0f || v > ds->images[oi]->height - 2.0f) continue;

            float pd = patch_distance_3x3(ds->images[oi], u, v, ref_rgb);
            float sc = score_from_patchdist(pd);
            if (sc > 0.0f) {
                support++;
                score += sc;

                float rgb[3];
                sample_bilinear_rgb(ds->images[oi], u, v, rgb);
                col_sum[0] += rgb[0];
                col_sum[1] += rgb[1];
                col_sum[2] += rgb[2];
                col_n++;
            }
        }

        float total = score + 0.35f * (float)support;
        if (support > best_support || (support == best_support && total > best_score)) {
            best_support = support;
            best_score = total;
            best_invz = invz;
            best_col[0] = col_sum[0] / (float)col_n;
            best_col[1] = col_sum[1] / (float)col_n;
            best_col[2] = col_sum[2] / (float)col_n;
        }
    }

    if (best_support < 2) return false;

    // refine
    float win = REFINE_WINDOW_FRAC * best_invz;
    float invz_lo = fmaxf(best_invz - win, INVDEPTH_FAR);
    float invz_hi = fminf(best_invz + win, INVDEPTH_NEAR);

    for (int s = 0; s < DEPTH_REFINE_SAMPLES; s++) {
        float a = (float)s / (float)(DEPTH_REFINE_SAMPLES - 1);
        float invz = invz_lo + a * (invz_hi - invz_lo);
        float depth = 1.0f / invz;

        float point[3] = {
            ray_o[0] + depth * ray_d[0],
            ray_o[1] + depth * ray_d[1],
            ray_o[2] + depth * ray_d[2]
        };

        int support = 0;
        float score = 0.0f;
        float col_sum[3] = {ref_rgb[0], ref_rgb[1], ref_rgb[2]};
        int col_n = 1;

        for (int ni = 0; ni < nbr_count; ni++) {
            int oi = nbr_ids[ni];
            float u, v, dcam;
            project_point_to_camera(point, ds->cameras[oi], &u, &v, &dcam);
            if (dcam <= 0.1f) continue;
            if (u < 1.0f || u > ds->images[oi]->width - 2.0f ||
                v < 1.0f || v > ds->images[oi]->height - 2.0f) continue;

            float pd = patch_distance_3x3(ds->images[oi], u, v, ref_rgb);
            float sc = score_from_patchdist(pd);
            if (sc > 0.0f) {
                support++;
                score += sc;

                float rgb[3];
                sample_bilinear_rgb(ds->images[oi], u, v, rgb);
                col_sum[0] += rgb[0];
                col_sum[1] += rgb[1];
                col_sum[2] += rgb[2];
                col_n++;
            }
        }

        float total = score + 0.35f * (float)support;
        if (support > best_support || (support == best_support && total > best_score)) {
            best_support = support;
            best_score = total;
            best_invz = invz;
            best_col[0] = col_sum[0] / (float)col_n;
            best_col[1] = col_sum[1] / (float)col_n;
            best_col[2] = col_sum[2] / (float)col_n;
        }
    }

    *out_depth = 1.0f / best_invz;
    out_color_avg[0] = best_col[0];
    out_color_avg[1] = best_col[1];
    out_color_avg[2] = best_col[2];
    *out_support = best_support;
    return true;
}

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }

static int compare_depth(const void *a, const void *b) {
    float da = ((const ProjectedGaussian*)a)->depth;
    float db = ((const ProjectedGaussian*)b)->depth;
    if (da < db) return 1;
    if (da > db) return -1;
    return 0;
}

static inline void project_gaussian(const Gaussian *g, const Camera *cam,
                                    float *mean2d, float *cov2d, float *depth) {
    float t[3];
    t[0] = g->pos[0] - cam->position[0];
    t[1] = g->pos[1] - cam->position[1];
    t[2] = g->pos[2] - cam->position[2];

    float cam_space[3];
    cam_space[0] = cam->rotation[0]*t[0] + cam->rotation[3]*t[1] + cam->rotation[6]*t[2];
    cam_space[1] = cam->rotation[1]*t[0] + cam->rotation[4]*t[1] + cam->rotation[7]*t[2];
    cam_space[2] = cam->rotation[2]*t[0] + cam->rotation[5]*t[1] + cam->rotation[8]*t[2];

    *depth = -cam_space[2];
    if (*depth <= 0.0f) return;

    float fx = cam->focal;
    float fy = cam->focal;

    float z_inv = 1.0f / (-cam_space[2]);
    mean2d[0] = fx * cam_space[0] * z_inv + cam->width * 0.5f;
    mean2d[1] = -fy * cam_space[1] * z_inv + cam->height * 0.5f;

    float cov3d[9];
    compute_covariance_3d(g, cov3d);

    // Cov_cam = R^T * Cov * R
    float temp[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) temp[i*3+j] += cam->rotation[k*3+i] * cov3d[k*3+j];
        }
    }

    float cov_cam[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov_cam[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) cov_cam[i*3+j] += temp[i*3+k] * cam->rotation[k*3+j];
        }
    }

    float z = -cam_space[2];
    float z2 = z * z;

    float J[6] = {
        fx / z, 0.0f, -fx * cam_space[0] / z2,
        0.0f, fy / z,  fy * cam_space[1] / z2
    };

    float temp2[6];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            temp2[i*3+j] = 0.0f;
            for (int k = 0; k < 3; k++) temp2[i*3+j] += J[i*3+k] * cov_cam[k*3+j];
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cov2d[i*2+j] = 0.0f;
            for (int k = 0; k < 3; k++) cov2d[i*2+j] += temp2[i*3+k] * J[j*3+k];
        }
    }

    // Stabilize + add render blur to reduce "point cloud" look.
    // (cov in px^2)
    float blur2 = RENDER_ADDITIONAL_SIGMA_PX * RENDER_ADDITIONAL_SIGMA_PX;
    cov2d[0] += 0.25f + blur2;
    cov2d[3] += 0.25f + blur2;
}

static void render_image(const GaussianModel *model, const Camera *cam,
                         unsigned char *image, int width, int height) {
    printf("  Rendering %dx%d\n", width, height);
    memset(image, 0, (size_t)width * height * 3);

    ProjectedGaussian *projected = (ProjectedGaussian*)malloc((size_t)model->num_gaussians * sizeof(ProjectedGaussian));
    int num_visible = 0;

    Camera render_cam = *cam;
    render_cam.width = width;
    render_cam.height = height;
    render_cam.focal = cam->focal * ((float)width / (float)cam->width);

    for (int i = 0; i < model->num_gaussians; i++) {
        const Gaussian *g = &model->gaussians[i];

        ProjectedGaussian pg;
        pg.index = i;
        pg.depth = 0.0f;

        project_gaussian(g, &render_cam, pg.mean2d, pg.cov2d, &pg.depth);
        if (pg.depth <= 0.1f) continue;

        if (pg.mean2d[0] < -80 || pg.mean2d[0] > width + 80 ||
            pg.mean2d[1] < -80 || pg.mean2d[1] > height + 80) continue;

        for (int c = 0; c < 3; c++) {
            pg.color[c] = SH_C0 * g->sh[0][c] + 0.5f;
            pg.color[c] = clampf(pg.color[c], 0.0f, 1.0f);
        }

        // a bit stronger alpha clamp for more solid surfaces
        pg.alpha = clampf(sigmoid(g->opacity), 0.0f, 0.9995f);

        projected[num_visible++] = pg;
    }

    printf("  Visible: %d/%d\n", num_visible, model->num_gaussians);
    if (num_visible == 0) { free(projected); return; }

    qsort(projected, (size_t)num_visible, sizeof(ProjectedGaussian), compare_depth);

    float *accum = (float*)calloc((size_t)width * height * 4, sizeof(float));
    for (int i = 0; i < width * height; i++) accum[i*4 + 3] = 1.0f; // transmittance

    for (int gi = 0; gi < num_visible; gi++) {
        const ProjectedGaussian *pg = &projected[gi];

        float a = pg->cov2d[0];
        float b = pg->cov2d[1];
        float c = pg->cov2d[3];
        float det = a*c - b*b;
        if (det <= 1e-10f) continue;

        float inv_det = 1.0f / det;
        float inv_cov00 =  c * inv_det;
        float inv_cov01 = -b * inv_det;
        float inv_cov11 =  a * inv_det;

        float radius = BBOX_SIGMAS * sqrtf(fmaxf(a, c));
        int x_min = (int)fmaxf(0.0f, floorf(pg->mean2d[0] - radius));
        int x_max = (int)fminf((float)(width - 1), ceilf(pg->mean2d[0] + radius));
        int y_min = (int)fmaxf(0.0f, floorf(pg->mean2d[1] - radius));
        int y_max = (int)fminf((float)(height - 1), ceilf(pg->mean2d[1] + radius));
        if (x_min > x_max || y_min > y_max) continue;

        for (int y = y_min; y <= y_max; y++) {
            for (int x = x_min; x <= x_max; x++) {
                float dx = (float)x - pg->mean2d[0];
                float dy = (float)y - pg->mean2d[1];

                float q = dx*(inv_cov00*dx + inv_cov01*dy) + dy*(inv_cov01*dx + inv_cov11*dy);
                float power = -0.5f * q;
                if (power < POWER_CUTOFF) continue;

                float weight = expf(power);
                float alpha = fminf(0.9995f, pg->alpha * weight);

                int idx = (y * width + x) * 4;
                float T = accum[idx + 3];
                if (T > 1e-4f) {
                    accum[idx + 0] += T * alpha * pg->color[0];
                    accum[idx + 1] += T * alpha * pg->color[1];
                    accum[idx + 2] += T * alpha * pg->color[2];
                    accum[idx + 3] *= (1.0f - alpha);
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < width * height; i++) {
        image[i*3 + 0] = (unsigned char)clampf(accum[i*4 + 0] * 255.0f, 0.0f, 255.0f);
        image[i*3 + 1] = (unsigned char)clampf(accum[i*4 + 1] * 255.0f, 0.0f, 255.0f);
        image[i*3 + 2] = (unsigned char)clampf(accum[i*4 + 2] * 255.0f, 0.0f, 255.0f);
    }

    free(accum);
    free(projected);
}

static GaussianModel* initialize_gaussians_from_dataset(Dataset *dataset) {
    GaussianModel *model = (GaussianModel*)malloc(sizeof(GaussianModel));
    model->gaussians = (Gaussian*)malloc((size_t)MAX_GAUSSIANS * sizeof(Gaussian));
    model->num_gaussians = 0;

    printf("Initializing Gaussians (patch consistency + inv-depth refine)...\n");

    int num_ref = dataset->num_images < REF_CAM_COUNT ? dataset->num_images : REF_CAM_COUNT;
    int cam_step = dataset->num_images / num_ref;
    if (cam_step < 1) cam_step = 1;

    for (int ref = 0; ref < num_ref && model->num_gaussians < MAX_GAUSSIANS; ref++) {
        int cam_idx = ref * cam_step;
        if (cam_idx >= dataset->num_images) cam_idx = dataset->num_images - 1;

        Image *ref_img = dataset->images[cam_idx];
        Camera *ref_cam = dataset->cameras[cam_idx];

        int nbr_ids[NEIGHBOR_CAM_COUNT];
        compute_nearest_cameras(dataset, cam_idx, nbr_ids, NEIGHBOR_CAM_COUNT);

        int total_pixels = ref_img->width * ref_img->height;
        int step = (int)fmaxf(3.0f, sqrtf((float)total_pixels / (float)INIT_SAMPLES_PER_REF));
        printf("Ref cam %d/%d idx=%d step=%d (img %dx%d)\n",
               ref+1, num_ref, cam_idx, step, ref_img->width, ref_img->height);

        int created_before = model->num_gaussians;

        for (int y = step/2; y < ref_img->height && model->num_gaussians < MAX_GAUSSIANS; y += step) {
            for (int x = step/2; x < ref_img->width && model->num_gaussians < MAX_GAUSSIANS; x += step) {
                int idx = (y * ref_img->width + x) * 4;
                unsigned char a = ref_img->data[idx + 3];
                if (a < 64) continue;

                float depth;
                float col_avg[3];
                int support = 0;

                if (!find_best_depth_multiview(dataset, cam_idx, x, y,
                                               nbr_ids, NEIGHBOR_CAM_COUNT,
                                               &depth, col_avg, &support)) {
                    continue;
                }
                if (support < MIN_VIEWS_CONSISTENT) continue;

                float ray_o[3], ray_d[3];
                generate_ray(ref_cam, x, y, ray_o, ray_d);

                Gaussian *ga = &model->gaussians[model->num_gaussians++];

                ga->pos[0] = ray_o[0] + depth * ray_d[0];
                ga->pos[1] = ray_o[1] + depth * ray_d[1];
                ga->pos[2] = ray_o[2] + depth * ray_d[2];

                // Choose world sigma so screen sigma ~= TARGET_SCREEN_SIGMA_PX
                float sigma_world = TARGET_SCREEN_SIGMA_PX * depth / ref_cam->focal;
                sigma_world = clampf(sigma_world, SCALE_MIN_WORLD, SCALE_MAX_WORLD);

                ga->scale[0] = sigma_world;
                ga->scale[1] = sigma_world;
                ga->scale[2] = sigma_world;

                ga->rot[0] = 1.0f; ga->rot[1] = 0.0f; ga->rot[2] = 0.0f; ga->rot[3] = 0.0f;

                // Stronger opacity for better solidity
                // base logit + support boost
                float op = 1.4f + 0.30f * (float)(support - MIN_VIEWS_CONSISTENT);
                ga->opacity = clampf(op, 0.8f, 5.0f);

                ga->sh[0][0] = (col_avg[0] - 0.5f) / SH_C0;
                ga->sh[0][1] = (col_avg[1] - 0.5f) / SH_C0;
                ga->sh[0][2] = (col_avg[2] - 0.5f) / SH_C0;

                for (int i = 1; i < 16; i++) {
                    ga->sh[i][0] = 0.0f;
                    ga->sh[i][1] = 0.0f;
                    ga->sh[i][2] = 0.0f;
                }
            }
        }

        int created = model->num_gaussians - created_before;
        printf("  Created %d gaussians (total %d)\n", created, model->num_gaussians);
    }

    printf("Initialized %d Gaussians\n", model->num_gaussians);
    return model;
}

int main(void) {
    srand(42);

    Dataset *dataset = load_dataset("data/transforms.json", "data", 100);
    if (!dataset || dataset->num_images == 0) {
        printf("Failed to load dataset\n");
        return 1;
    }

    GaussianModel *model = initialize_gaussians_from_dataset(dataset);
    if (!model || model->num_gaussians == 0) {
        printf("No gaussians created.\n");
        return 1;
    }

    printf("\nRendering views...\n");

    int render_width = 400;
    int render_height = 400;
    unsigned char *render_buffer = (unsigned char*)malloc((size_t)render_width * render_height * 3);

    for (int i = 0; i < dataset->num_images; i += 10) {
        printf("\nTrain view %d:\n", i);
        render_image(model, dataset->cameras[i], render_buffer, render_width, render_height);

        char filename[256];
        snprintf(filename, sizeof(filename), "train_%d_splat.png", i);
        save_png(filename, render_buffer, render_width, render_height);
    }

    if (dataset->num_images >= 2) {
        Camera interp_cam;
        interp_cam.width = render_width;
        interp_cam.height = render_height;

        for (int i = 0; i < 5; i++) {
            printf("\nNovel view %d:\n", i);
            float alpha = i / 4.0f;
            interpolate_cameras(dataset->cameras[0], dataset->cameras[dataset->num_images - 1],
                                alpha, &interp_cam);

            render_image(model, &interp_cam, render_buffer, render_width, render_height);

            char filename[256];
            snprintf(filename, sizeof(filename), "novel_%d_splat.png", i);
            save_png(filename, render_buffer, render_width, render_height);
        }
    }

    printf("\nDone!\n");

    free(render_buffer);
    free(model->gaussians);
    free(model);
    free_dataset(dataset);
    return 0;
}