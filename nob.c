#define NOB_IMPLEMENTATION
#include "nob.h"

#define BUILD_FOLDER "build/"

int main(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);
    nob_shift(argv, argc);
    
    if (!nob_mkdir_if_not_exists("build")) return 1;

    Nob_Cmd cmd = {0};
    nob_cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-O2", "-g");
    nob_cmd_append(&cmd, "-I/usr/include/gdal");
    nob_cmd_append(&cmd, "-Ivendor/onnxruntime/include");
    nob_cmd_append(&cmd, "-o", BUILD_FOLDER"swath");
    nob_cmd_append(&cmd, "src/main.c");
    nob_cmd_append(&cmd, "-lproj", "-lgdal", "-lm");
    nob_cmd_append(&cmd, "-Lvendor/onnxruntime/lib", "-lonnxruntime");
    nob_cmd_append(&cmd, "-Wl,-rpath,$ORIGIN/../vendor/onnxruntime/lib");

    if (!nob_cmd_run_sync(cmd)) return 1;

    if (argc > 0 && strcmp(nob_shift(argv, argc), "run") == 0) {
	Nob_Cmd run = {0};
	nob_cmd_append(&run, BUILD_FOLDER"swath");
	while (argc > 0) nob_cmd_append(&run, nob_shift(argv, argc));
	if (!nob_cmd_run_sync(run)) return 1;
    }

    return 0;
}
