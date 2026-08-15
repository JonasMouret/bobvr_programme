/*
 * GoPro MAX (.360) -> equirectangular, in a single OpenCL pass.
 *
 * The .360 container holds two HEVC tracks of SRC_W x SRC_H. Together they
 * carry the six faces of an equi-angular cubemap (EAC), but not as a plain
 * grid: the two lenses overlap, and each track buries a stitch seam in the
 * middle of its first and third face.
 *
 *   track 0:  [ LEFT  |seam|  ][    FRONT     ][ RIGHT |seam|  ]
 *   track 1:  [ DOWN  |seam|  ][    BACK      ][  UP   |seam|  ]
 *
 * This kernel walks backwards: for each output pixel it derives a viewing
 * direction, finds the cube face that direction hits, converts to EAC face
 * coordinates, undoes the seam packing, and samples the source track. Doing
 * it in one pass keeps the whole transform on the GPU -- no intermediate EAC
 * frame is ever materialised.
 *
 * The caller prepends a block of #defines (see geometry.py) so the same
 * source adapts to different capture resolutions and orientations.
 *
 * Invoked once per plane by ffmpeg's program_opencl filter, so every
 * coordinate is normalised and plane dimensions are read at runtime; chroma
 * planes therefore need no special casing.
 */

__constant sampler_t smp_linear = CLK_NORMALIZED_COORDS_TRUE |
                                  CLK_ADDRESS_CLAMP_TO_EDGE |
                                  CLK_FILTER_LINEAR;

__constant sampler_t smp_nearest = CLK_NORMALIZED_COORDS_FALSE |
                                   CLK_ADDRESS_CLAMP_TO_EDGE |
                                   CLK_FILTER_NEAREST;

/* Cube faces, matching the order used to derive the layout constants. */
#define FACE_FRONT 0
#define FACE_BACK  1
#define FACE_LEFT  2
#define FACE_RIGHT 3
#define FACE_UP    4
#define FACE_DOWN  5

/* ---------------------------------------------------------------- sampling */

static float sample_plane(__read_only image2d_t img, float2 xy_luma)
{
    /* Geometry is computed in luma pixels; rescale to this plane's size. */
    int2 dim = get_image_dim(img);
    float2 norm = (float2)(xy_luma.x * (1.0f / (float)SRC_W),
                           xy_luma.y * (1.0f / (float)SRC_H));

#if INTERP_CUBIC
    /* Catmull-Rom. Sharper than bilinear where the projection magnifies the
     * source, which is most of the frame away from the equator. */
    float2 pix = (float2)(norm.x * dim.x, norm.y * dim.y) - 0.5f;
    float2 base = floor(pix);
    float2 f = pix - base;

    float wx[4], wy[4];
    float t = f.x, t2 = t * t, t3 = t2 * t;
    wx[0] = 0.5f * (-t3 + 2.0f * t2 - t);
    wx[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
    wx[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
    wx[3] = 0.5f * (t3 - t2);
    t = f.y; t2 = t * t; t3 = t2 * t;
    wy[0] = 0.5f * (-t3 + 2.0f * t2 - t);
    wy[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
    wy[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
    wy[3] = 0.5f * (t3 - t2);

    float acc = 0.0f;
    for (int j = 0; j < 4; j++) {
        float row = 0.0f;
        for (int i = 0; i < 4; i++)
            row += wx[i] * read_imagef(img, smp_nearest,
                                       (float2)(base.x + i - 1 + 0.5f,
                                                base.y + j - 1 + 0.5f)).x;
        acc += wy[j] * row;
    }
    return acc;
#else
    return read_imagef(img, smp_linear, norm).x;
#endif
}

/*
 * Map an x coordinate on the seamless EAC row to the packed source track.
 *
 * Outside the seams this is a plain shift: the source is SEAM_TOTAL pixels
 * wider than the EAC row it encodes. Inside a seam the source keeps both
 * lenses' pixels, so we cross-fade them exactly as GoPro's own stitch does --
 * that blend is the only place the two lenses are reconciled, and skipping it
 * leaves a visible edge down the frame.
 */
static float sample_row(__read_only image2d_t img, float ex, float ey)
{
    if (ex < SEAM0_EAC) {
        return sample_plane(img, (float2)(ex, ey));
    } else if (ex < SEAM0_EAC + SEAM_EAC_W) {
        float s = (ex - SEAM0_EAC) * ((float)SEAM_HALF / (float)SEAM_EAC_W);
        float w = (s + 1.0f) * (1.0f / ((float)SEAM_HALF + 1.0f));
        float lo = sample_plane(img, (float2)(SEAM0_SRC + s, ey));
        float hi = sample_plane(img, (float2)(SEAM0_SRC + SEAM_HALF + s, ey));
        return mix(lo, hi, w);
    } else if (ex < SEAM1_EAC) {
        return sample_plane(img, (float2)(ex + SEAM_HALF / 2.0f, ey));
    } else if (ex < SEAM1_EAC + SEAM_EAC_W) {
        float s = (ex - SEAM1_EAC) * ((float)SEAM_HALF / (float)SEAM_EAC_W);
        float w = (s + 1.0f) * (1.0f / ((float)SEAM_HALF + 1.0f));
        float lo = sample_plane(img, (float2)(SEAM1_SRC + s, ey));
        float hi = sample_plane(img, (float2)(SEAM1_SRC + SEAM_HALF + s, ey));
        return mix(lo, hi, w);
    } else {
        return sample_plane(img, (float2)(ex + (float)SEAM_HALF, ey));
    }
}

/* ---------------------------------------------------------------- geometry */

__kernel void gopromax_equirect(__write_only image2d_t dst,
                                unsigned int index,
                                __read_only image2d_t track0,
                                __read_only image2d_t track1)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 dim = get_image_dim(dst);
    if (p.x >= dim.x || p.y >= dim.y)
        return;

    /* Equirectangular: x spans a full turn, y spans pole to pole. */
    float lon = ((2.0f * (p.x + 0.5f)) / dim.x - 1.0f) * M_PI_F;
    float lat = ((2.0f * (p.y + 0.5f)) / dim.y - 1.0f) * M_PI_F * 0.5f;

    float cos_lat = cos(lat);
    float3 v = (float3)(cos_lat * sin(lon), sin(lat), cos_lat * cos(lon));

#if APPLY_VIEW_SCALE
    /*
     * Widen (or tighten) what a player shows when the file opens.
     *
     * No metadata field carries a field of view for spherical video, so the
     * only way to decide it is to bake it into the sphere. A player draws a
     * rectilinear view: a point at angle b from where it looks lands on screen
     * at tan(b), scaled by its own fixed field of view. So to make its window
     * hold a wider view -- and hold it the way a wider lens would, with
     * straight lines still straight -- the scene has to be pulled outwards in
     * *tangent* space, not in angle:
     *
     *     tan(scene angle) = VIEW_SCALE_C * tan(angle in this file)
     *
     * The player's own projection then cancels the tangents exactly, and what
     * it draws is a true rectilinear view of the wider angle. Pulling on the
     * angles instead (the obvious thing) leaves tangents in the composition
     * and bends every straight edge into a curve.
     *
     * Written as an atan2 so it holds all the way round: the transform turns
     * about the viewing axis only, keeps 0, 90 and 180 degrees where they are,
     * and stays ordered for any positive C -- the sphere is redistributed, not
     * cut. What the front gains, the far side gives up in sharpness.
     */
    float r = length(v.xy);              /* sin of the angle off-axis */
    if (r > 1e-6f) {
        float beta = atan2(VIEW_SCALE_C * r, v.z);
        float s = sin(beta) / r;
        v = (float3)(v.x * s, v.y * s, cos(beta));
    }
#endif

#if APPLY_ROTATION
    /* yaw about Y, then pitch about X, then roll about Z. */
    float cy = cos(YAW_RAD),   sy = sin(YAW_RAD);
    float cp = cos(PITCH_RAD), sp = sin(PITCH_RAD);
    float cr = cos(ROLL_RAD),  sr = sin(ROLL_RAD);
    float3 t;
    t.x = cy * v.x + sy * v.z;
    t.y = v.y;
    t.z = -sy * v.x + cy * v.z;
    v = t;
    t.x = v.x;
    t.y = cp * v.y - sp * v.z;
    t.z = sp * v.y + cp * v.z;
    v = t;
    t.x = cr * v.x - sr * v.y;
    t.y = sr * v.x + cr * v.y;
    t.z = v.z;
    v = t;
#endif

    /* Which cube face does this direction hit?  Mirrors ffmpeg's v360 so the
     * output stays interchangeable with the reference implementation. */
    float azim = atan2(v.x, v.z);
    float elev = asin(clamp(v.y, -1.0f, 1.0f));

    int face;
    float azim_norm;
    if (azim >= -M_PI_F / 4.0f && azim < M_PI_F / 4.0f) {
        face = FACE_FRONT; azim_norm = azim;
    } else if (azim >= -3.0f * M_PI_F / 4.0f && azim < -M_PI_F / 4.0f) {
        face = FACE_LEFT;  azim_norm = azim + M_PI_F / 2.0f;
    } else if (azim >= M_PI_F / 4.0f && azim < 3.0f * M_PI_F / 4.0f) {
        face = FACE_RIGHT; azim_norm = azim - M_PI_F / 2.0f;
    } else {
        face = FACE_BACK;  azim_norm = azim + (azim > 0.0f ? -M_PI_F : M_PI_F);
    }

    float elev_limit = atan(cos(azim_norm));
    if (elev > elev_limit)
        face = FACE_DOWN;
    else if (elev < -elev_limit)
        face = FACE_UP;

    /* Gnomonic coordinates on that face, in [-1, 1]. */
    float uf, vf;
    switch (face) {
    case FACE_RIGHT: uf =  v.z / v.x; vf =  v.y / v.x; break;
    case FACE_LEFT:  uf =  v.z / v.x; vf = -v.y / v.x; break;
    case FACE_UP:    uf = -v.x / v.y; vf = -v.z / v.y; break;
    case FACE_DOWN:  uf =  v.x / v.y; vf = -v.z / v.y; break;
    case FACE_FRONT: uf =  v.x / v.z; vf =  v.y / v.z; break;
    default:         uf =  v.x / v.z; vf = -v.y / v.z; break; /* BACK */
    }

    /* Equi-angular: the cubemap is sampled uniformly in angle, not in
     * tangent, which is what buys EAC its even pixel density. */
    float a = M_2_PI_F * atan(uf) + 0.5f;
    float b = M_2_PI_F * atan(vf) + 0.5f;

    /* Per-face placement in the 3x2 grid, including each face's stored
     * orientation. Derived empirically from v360, not assumed. */
    float fa, fb;
    int col, row;
    switch (face) {
    case FACE_LEFT:  fa = 1.0f - a; fb = b;        col = 0; row = 0; break;
    case FACE_FRONT: fa = a;        fb = b;        col = 1; row = 0; break;
    case FACE_RIGHT: fa = 1.0f - a; fb = b;        col = 2; row = 0; break;
    case FACE_DOWN:  fa = b;        fb = 1.0f - a; col = 0; row = 1; break;
    case FACE_BACK:  fa = 1.0f - b; fb = a;        col = 1; row = 1; break;
    default:         fa = b;        fb = 1.0f - a; col = 2; row = 1; break; /* UP */
    }

    float ex = (fa + col) * (float)EAC_FACE;
    float ey = fb * (float)EAC_FACE;

    float val = (row == 0) ? sample_row(track0, ex, ey)
                           : sample_row(track1, ex, ey);

    write_imagef(dst, p, (float4)(val, 0.0f, 0.0f, 1.0f));
}
