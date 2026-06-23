#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <proj.h>
#include <gdal.h>
#include <onnxruntime_c_api.h>


#define R 6378137.0
#define DEG2RAD (M_PI / 180.0)

#define ORT_CHECK(g, expr) do {                               \
    OrtStatus *_st = (expr);                                  \
    if (_st) {                                                \
    fprintf(stderr, "onnx: %s\n", (g)->GetErrorMessage(_st)); \
    (g)->ReleaseStatus(_st);                                  \
    return 1;                                                 \
    }                                                         \
} while (0)                                                   \

static void webmerc_fwd(double lon, double lat, double *x, double *y)
{
    *x = R * (lon * DEG2RAD);
    *y = R * log(tan(M_PI / 4.0 + (lat * DEG2RAD) / 2.0));
}

static int cmd_proj(int argc, char **argv)
{
    if (argc != 2) {
    fprintf(stderr, "usage: swath <lon> <lat>\n");
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
    if (argc != 1) {
    fprintf(stderr, "usage: swath info <raster.tif>\n");
    return 1;
    }

    const char* path = argv[0];

    GDALAllRegister();

    GDALDatasetH ds = GDALOpen(path, GA_ReadOnly);
    if (!ds) {
    fprintf(stderr, "GDALOpen failed: %s\n", path);
    return 1;
    }

    int nx = GDALGetRasterXSize(ds);
    int ny = GDALGetRasterYSize(ds);
    int nb = GDALGetRasterCount(ds);
    printf("size %d x %d, band %d\n", nx, ny, nb);

    double gt[6];
    if (GDALGetGeoTransform(ds, gt) == CE_None) {
    printf("geotransform %.10g %.10g %.10g %.10g %.10g %.10g\n",
    gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]);
    }

    GDALRasterBandH band = GDALGetRasterBand(ds, 1);
    GDALDataType dt = GDALGetRasterDataType(band);
    int has_nodata = 0;
    double nodata = GDALGetRasterNoDataValue(band, &has_nodata);
    if (has_nodata) {
    printf("band1 type %s, nodata %.0f\n", GDALGetDataTypeName(dt), nodata);
    } else {
    printf("band1 type %s, nodata none\n", GDALGetDataTypeName(dt));
    }

    double v;
    if (GDALRasterIO(band, GF_Read, 0, 0, 1, 1, &v, 1, 1, GDT_Float64, 0, 0) == CE_None) {
    printf("pixel(0, 0) %.3f\n", v);
    }

    GDALClose(ds);
    return 0;
}

static int cmd_stats(int argc, char **argv)
{
    if (argc != 1) {
    fprintf(stderr, "usage: swath stats <raster.tif>\n");
    return 1;
    }
    const char *path = argv[0];

    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(path, GA_ReadOnly);
    if (!ds) {
    fprintf(stderr, "GDALOpen failed: %s\n", path);
    return 1;
    }

    int nx = GDALGetRasterXSize(ds);
    int ny = GDALGetRasterYSize(ds);
    GDALRasterBandH band = GDALGetRasterBand(ds, 1);

    int has_nodata = 0;
    double nodata = GDALGetRasterNoDataValue(band, &has_nodata);

    size_t n = (size_t)nx * ny;
    float *buf = malloc(n * sizeof(float));

    if (!buf) {
    fprintf(stderr, "oom: %zu floats\n", n);
    GDALClose(ds);
    return 1;
    }

    if (GDALRasterIO(band, GF_Read, 0, 0, nx, ny, buf, nx, ny, GDT_Float32, 0, 0) != CE_None) {
    fprintf(stderr, "RasterIO failed\n");
    free(buf);
    GDALClose(ds);
    return 1;
    }

    double sum = 0.0, mn = 0.0, mx = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; i++) {
    double val = buf[i];
    if (has_nodata && val == nodata) continue;
    if (valid == 0) { mn = mx = val; }
    else { if (val < mn) mn = val; if (val > mx) mx = val; }
    sum += val;
    valid++;
    }

    double mean = valid ? sum / valid : 0.0;
    printf("pixels %zu, valid %zu\n", n, valid);
    printf("min %.3f max %.3f mean %.3f\n", mn, mx, mean);

    free(buf);
    GDALClose(ds);
    return 0;
}

static int cmd_mask(int argc, char **argv)
{
    if (argc != 3) {
    fprintf(stderr, "usage: swath mask <in.tif> <out.tif> <threshold>\n");
    return 1;
    }
    const char *in_path  = argv[0];
    const char *out_path = argv[1];
    double thresh = atof(argv[2]);

    GDALAllRegister();

    GDALDatasetH in = GDALOpen(in_path, GA_ReadOnly);
    if (!in) { fprintf(stderr, "open failed: %s\n", in_path); return 1; }

    int nx = GDALGetRasterXSize(in);
    int ny = GDALGetRasterYSize(in);
    GDALRasterBandH iband = GDALGetRasterBand(in, 1);

    int has_nodata = 0;
    double nodata = GDALGetRasterNoDataValue(iband, &has_nodata);

    size_t n = (size_t)nx * ny;
    float *buf = malloc(n * sizeof(float));
    unsigned char *mask = malloc(n);
    if (!buf || !mask) { fprintf(stderr, "oom\n"); return 1; }

    if (GDALRasterIO(iband, GF_Read, 0, 0, nx, ny, buf, nx, ny, GDT_Float32, 0, 0) != CE_None) {
    fprintf(stderr, "read failed\n"); return 1;
    }

    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
    if (has_nodata && buf[i] == nodata) { mask[i] = 0; continue; }
    /* ONNX forward pass should be here */
    mask[i] = (buf[i] > thresh) ? 1 : 0;
    hits += mask[i];
    }

    /* --- write --- */
    GDALDriverH drv  = GDALGetDriverByName("GTiff");
    GDALDatasetH out = GDALCreate(drv, out_path, nx, ny, 1, GDT_Byte, NULL);
    if (!out) { fprintf(stderr, "create failed: %s\n", out_path); return 1; }

    double gt[6];
    GDALGetGeoTransform(in, gt);
    GDALSetGeoTransform(out, gt);                      /* same grid->world */
    GDALSetProjection(out, GDALGetProjectionRef(in));  /* same CRS (WKT)   */

    GDALRasterBandH oband = GDALGetRasterBand(out, 1);
    if (GDALRasterIO(oband, GF_Write, 0, 0, nx, ny, mask, nx, ny, GDT_Byte, 0, 0) != CE_None) {
    fprintf(stderr, "write failed\n"); return 1;
    }

    GDALClose(out);
    GDALClose(in);
    free(buf);
    free(mask);

    printf("wrote %s: %zu/%zu pixels above %.1f\n", out_path, hits, n, thresh);
    return 0;
}

static int cmd_infer(int argc, char **argv)
{
    if (argc != 1) {
    fprintf(stderr, "usage: swath infer <model.onnx>\n");
    return 1;
    }
    const char *model_path = argv[0];

    const OrtApi *g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g) { fprintf(stderr, "onnx: api version mismatch\n"); return 1; }

    OrtEnv *env = NULL;
    ORT_CHECK(g, g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "swath", &env));

    OrtSessionOptions *opts = NULL;
    ORT_CHECK(g, g->CreateSessionOptions(&opts));

    OrtSession *session = NULL;
    ORT_CHECK(g, g->CreateSession(env, model_path, opts, &session));

    OrtMemoryInfo *mem = NULL;
    ORT_CHECK(g, g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));

    float   in[6]    = {1, 2, 3, 4, 5, 6};
    int64_t shape[4] = {1, 1, 2, 3};

    OrtValue *input_tensor = NULL;
    ORT_CHECK(g, g->CreateTensorWithDataAsOrtValue(
    mem, in, sizeof(in), shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char *input_names[]  = { "input" };
    const char *output_names[] = { "output" };

    OrtValue *output_tensor = NULL;
    ORT_CHECK(g, g->Run(session, NULL, input_names, (const OrtValue *const *)&input_tensor, 1, output_names, 1, &output_tensor));

    float *out = NULL;
    ORT_CHECK(g, g->GetTensorMutableData(output_tensor, (void **)&out));

    int ok = 1;
    for (int i = 0; i < 6; i++) {
    printf("in %.1f  out %.1f  expect %.1f\n", in[i], out[i], in[i] * 2.0f);
    if(out[i] != in[i] * 2.0f) ok = 0;
    }
    printf("%s\n", ok ? "PLUMBING OK" : "MISMATCH");

    g->ReleaseValue(output_tensor);
    g->ReleaseValue(input_tensor);
    g->ReleaseMemoryInfo(mem);
    g->ReleaseSession(session);
    g->ReleaseSessionOptions(opts);
    g->ReleaseEnv(env);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
    fprintf(stderr, "usage: swath <proj|info> ...\n");
    return 1;
    }
    const char *sub = argv[1];
    if (strcmp(sub, "proj")  == 0) return cmd_proj(argc - 2, argv + 2);
    if (strcmp(sub, "info")  == 0) return cmd_info(argc - 2, argv + 2);
    if (strcmp(sub, "stats") == 0) return cmd_stats(argc - 2, argv + 2);
    if (strcmp(sub, "mask")  == 0) return cmd_mask(argc - 2, argv + 2);
    if (strcmp(sub, "infer") == 0) return cmd_infer(argc - 2, argv + 2);

    fprintf(stderr, "unknown subcommand: %s\n", sub);
    return 1;
}
