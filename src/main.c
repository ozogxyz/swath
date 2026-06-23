#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <proj.h>
#include <gdal.h>
#include <onnxruntime_c_api.h>


#define R 6378137.0
#define DEG2RAD (M_PI / 180.0)

#define ORT_CHECK(g, expr) do {                                   \
    OrtStatus *_st = (expr);                                      \
    if (_st) {                                                    \
        fprintf(stderr, "onnx: %s\n", (g)->GetErrorMessage(_st)); \
        (g)->ReleaseStatus(_st);                                  \
        return 1;                                                 \
    }                                                             \
} while (0)

static void webmerc_fwd(double lon, double lat, double *x, double *y)
{
    *x = R * (lon * DEG2RAD);
    *y = R * log(tan(M_PI / 4.0 + (lat * DEG2RAD) / 2.0));
}

typedef struct {
    GDALDatasetH    ds;
    int             nx, ny, nbands;
    double          gt[6];
    const char     *wkt;
    GDALRasterBandH band;        /* band 1 */
    int             has_nodata;
    double          nodata;
} Raster;

static int raster_open(Raster *r, const char *path)
{
    r->ds = GDALOpen(path, GA_ReadOnly);
    if (!r->ds) { fprintf(stderr, "open failed: %s\n", path); return 1; }
    r->nx     = GDALGetRasterXSize(r->ds);
    r->ny     = GDALGetRasterYSize(r->ds);
    r->nbands = GDALGetRasterCount(r->ds);
    GDALGetGeoTransform(r->ds, r->gt);
    r->wkt    = GDALGetProjectionRef(r->ds);
    r->band   = GDALGetRasterBand(r->ds, 1);
    r->nodata = GDALGetRasterNoDataValue(r->band, &r->has_nodata);
    return 0;
}

static float *raster_read_f32(const Raster *r)   /* band 1, full extent; caller frees */
{
    size_t n = (size_t)r->nx * r->ny;
    float *buf = malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "oom: %zu floats\n", n); return NULL; }
    if (GDALRasterIO(r->band, GF_Read, 0, 0, r->nx, r->ny,
                     buf, r->nx, r->ny, GDT_Float32, 0, 0) != CE_None) {
        fprintf(stderr, "read failed\n");
        free(buf);
        return NULL;
    }
    return buf;
}

static int raster_write_byte(const Raster *like, const char *path, const unsigned char *data)
{
    GDALDriverH  drv = GDALGetDriverByName("GTiff");
    GDALDatasetH out = GDALCreate(drv, path, like->nx, like->ny, 1, GDT_Byte, NULL);
    if (!out) { fprintf(stderr, "create failed: %s\n", path); return 1; }

    GDALSetGeoTransform(out, (double *)like->gt);   /* copy grid->world */
    GDALSetProjection(out, like->wkt);              /* copy CRS         */

    GDALRasterBandH b = GDALGetRasterBand(out, 1);
    CPLErr rc = GDALRasterIO(b, GF_Write, 0, 0, like->nx, like->ny,
                             (void *)data, like->nx, like->ny, GDT_Byte, 0, 0);
    GDALClose(out);                                 /* flushes */
    if (rc != CE_None) { fprintf(stderr, "write failed\n"); return 1; }
    return 0;
}

static void raster_close(Raster *r) { if (r->ds) GDALClose(r->ds); }

/* ---- Model: the four ORT handles + api pointer the cmds kept rebuilding ---- */
typedef struct {
    const OrtApi      *g;
    OrtEnv            *env;
    OrtSessionOptions *opts;
    OrtSession        *session;
    OrtMemoryInfo     *mem;
} Model;

static int model_open(Model *m, const char *path)
{
    m->g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!m->g) { fprintf(stderr, "onnx: api version mismatch\n"); return 1; }
    const OrtApi *g = m->g;
    m->env = NULL; m->opts = NULL; m->session = NULL; m->mem = NULL;
    ORT_CHECK(g, g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "swath", &m->env));
    ORT_CHECK(g, g->CreateSessionOptions(&m->opts));
    ORT_CHECK(g, g->CreateSession(m->env, path, m->opts, &m->session));
    ORT_CHECK(g, g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem));
    return 0;
}

/* wraps `in` (no copy), runs, copies output into `out` (out_n elems) */
static int model_run(Model *m, const float *in, const int64_t *shape, size_t ndim,
                     float *out, size_t out_n)
{
    const OrtApi *g = m->g;

    size_t in_n = 1;
    for (size_t i = 0; i < ndim; i++) in_n *= (size_t)shape[i];

    OrtValue *it = NULL;
    ORT_CHECK(g, g->CreateTensorWithDataAsOrtValue(
        m->mem, (void *)in, in_n * sizeof(float),
        shape, ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &it));

    const char *in_names[]  = { "input" };
    const char *out_names[] = { "output" };
    OrtValue *ot = NULL;
    ORT_CHECK(g, g->Run(m->session, NULL,
        in_names, (const OrtValue *const *)&it, 1,
        out_names, 1, &ot));

    float *p = NULL;
    ORT_CHECK(g, g->GetTensorMutableData(ot, (void **)&p));

    memcpy(out, p, out_n * sizeof(float));
    g->ReleaseValue(ot);
    g->ReleaseValue(it);
    return 0;
}

static void model_close(Model *m)
{
    const OrtApi *g = m->g;
    if (m->mem)     g->ReleaseMemoryInfo(m->mem);
    if (m->session) g->ReleaseSession(m->session);
    if (m->opts)    g->ReleaseSessionOptions(m->opts);
    if (m->env)     g->ReleaseEnv(m->env);
}

static int cmd_proj(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: swath proj <lon> <lat>\n");
        return 1;
    }

    double lon = atof(argv[0]);
    double lat = atof(argv[1]);

    PJ_CONTEXT *C = proj_context_create();

    PJ *P = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:3857", NULL);
    if (!P) {
        fprintf(stderr, "crs_to_crs failed %s\n", proj_context_errno_string(C, proj_context_errno(C)));
        return 1;
    }

    /* rewire  to lon, lat / easting, northing human order */
    PJ *Pn = proj_normalize_for_visualization(C, P);
    proj_destroy(P);
    P = Pn;

    PJ_COORD a = proj_coord(lon, lat, 0, 0); /* lon first via normalize */
    PJ_COORD b = proj_trans(P, PJ_FWD, a);

    double hx, hy;
    webmerc_fwd(lon, lat, &hx, &hy);

    printf("proj %.3f %.3f\n", b.xy.x, b.xy.y);
    printf("hand %.3f %.3f\n", hx, hy);

    proj_destroy(P);
    proj_context_destroy(C);
    return 0;
}

static int cmd_info(int argc, char **argv)
{
    if (argc != 1) { fprintf(stderr, "usage: swath info <raster.tif>\n"); return 1; }

    GDALAllRegister();
    Raster r;
    if (raster_open(&r, argv[0])) return 1;

    printf("size %d x %d, bands %d\n", r.nx, r.ny, r.nbands);
    printf("geotransform %.10g %.10g %.10g %.10g %.10g %.10g\n",
           r.gt[0], r.gt[1], r.gt[2], r.gt[3], r.gt[4], r.gt[5]);

    GDALDataType dt = GDALGetRasterDataType(r.band);
    if (r.has_nodata) printf("band1 type %s, nodata %.0f\n", GDALGetDataTypeName(dt), r.nodata);
    else              printf("band1 type %s, nodata none\n", GDALGetDataTypeName(dt));

    double v;
    if (GDALRasterIO(r.band, GF_Read, 0, 0, 1, 1, &v, 1, 1, GDT_Float64, 0, 0) == CE_None)
        printf("pixel(0,0) %.3f\n", v);

    raster_close(&r);
    return 0;
}

static int cmd_stats(int argc, char **argv)
{
    if (argc != 1) { fprintf(stderr, "usage: swath stats <raster.tif>\n"); return 1; }

    GDALAllRegister();
    Raster r;
    if (raster_open(&r, argv[0])) return 1;

    float *buf = raster_read_f32(&r);
    if (!buf) return 1;

    size_t n = (size_t)r.nx * r.ny;
    double sum = 0.0, mn = 0.0, mx = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; i++) {
        double val = buf[i];
        if (r.has_nodata && val == r.nodata) continue;
        if (valid == 0) { mn = mx = val; }
        else { if (val < mn) mn = val; if (val > mx) mx = val; }
        sum += val;
        valid++;
    }

    double mean = valid ? sum / valid : 0.0;
    printf("pixels %zu, valid %zu\n", n, valid);
    printf("min %.3f max %.3f mean %.3f\n", mn, mx, mean);

    free(buf);
    raster_close(&r);
    return 0;
}

static int cmd_mask(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: swath mask <in.tif> <out.tif> <threshold>\n"); return 1; }
    double thresh = atof(argv[2]);

    GDALAllRegister();
    Raster r;
    if (raster_open(&r, argv[0])) return 1;

    float *buf = raster_read_f32(&r);
    if (!buf) return 1;

    size_t n = (size_t)r.nx * r.ny;
    unsigned char *mask = malloc(n);
    if (!mask) { fprintf(stderr, "oom\n"); return 1; }

    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (r.has_nodata && buf[i] == r.nodata) { mask[i] = 0; continue; }
        mask[i] = (buf[i] > thresh) ? 1 : 0;
        hits += mask[i];
    }

    if (raster_write_byte(&r, argv[1], mask)) return 1;

    free(buf); free(mask);
    raster_close(&r);
    printf("wrote %s: %zu/%zu pixels above %.1f\n", argv[1], hits, n, thresh);
    return 0;
}

static int cmd_infer(int argc, char **argv)
{
    if (argc != 1) { fprintf(stderr, "usage: swath infer <model.onnx>\n"); return 1; }

    Model m;
    if (model_open(&m, argv[0])) return 1;

    float   in[6]    = {1, 2, 3, 4, 5, 6};
    float   out[6];
    int64_t shape[4] = {1, 1, 2, 3};
    if (model_run(&m, in, shape, 4, out, 6)) return 1;

    int ok = 1;
    for (int i = 0; i < 6; i++) {
        printf("in %.1f  out %.1f  expect %.1f\n", in[i], out[i], in[i] * 2.0f);
        if (out[i] != in[i] * 2.0f) ok = 0;
    }
    printf("%s\n", ok ? "PLUMBING OK" : "MISMATCH");

    model_close(&m);
    return ok ? 0 : 1;
}

static int cmd_segment(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: swath segment <in.tif> <out.tif> <model.onnx>\n"); return 1; }

    GDALAllRegister();
    Raster r;
    if (raster_open(&r, argv[0])) return 1;

    float *buf = raster_read_f32(&r);
    if (!buf) return 1;

    Model m;
    if (model_open(&m, argv[2])) return 1;

    size_t n = (size_t)r.nx * r.ny;
    float         *out  = malloc(n * sizeof(float));
    unsigned char *mask = malloc(n);
    if (!out || !mask) { fprintf(stderr, "oom\n"); return 1; }

    int64_t shape[4] = {1, 1, r.ny, r.nx};
    if (model_run(&m, buf, shape, 4, out, n)) return 1;

    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (r.has_nodata && buf[i] == r.nodata) { mask[i] = 0; continue; }
        mask[i] = (out[i] > 0.5f) ? 1 : 0;
        hits += mask[i];
    }

    if (raster_write_byte(&r, argv[1], mask)) return 1;

    model_close(&m);
    raster_close(&r);
    free(buf); free(out); free(mask);
    printf("wrote %s: %zu/%zu foreground\n", argv[1], hits, n);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: swath <proj|info|stats|mask|infer|segment> ...\n");
        return 1;
    }
    const char *sub = argv[1];
    if (strcmp(sub, "proj")    == 0) return cmd_proj(argc - 2, argv + 2);
    if (strcmp(sub, "info")    == 0) return cmd_info(argc - 2, argv + 2);
    if (strcmp(sub, "stats")   == 0) return cmd_stats(argc - 2, argv + 2);
    if (strcmp(sub, "mask")    == 0) return cmd_mask(argc - 2, argv + 2);
    if (strcmp(sub, "infer")   == 0) return cmd_infer(argc - 2, argv + 2);
    if (strcmp(sub, "segment") == 0) return cmd_segment(argc - 2, argv + 2);

    fprintf(stderr, "unknown subcommand: %s\n", sub);
    return 1;
}
