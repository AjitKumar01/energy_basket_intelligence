#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

/* macOS sandbox interposer redirecting POSIX shm_open/shm_unlink (used by libomp on startup)
 * to /tmp/shm_* files when direct mach shared memory is blocked by sandbox permissions. */
int my_shm_open(const char *name, int oflag, ...) {
    char path[256];
    const char *p = (name[0] == '/') ? name + 1 : name;
    snprintf(path, sizeof(path), "/tmp/shm_%s", p);
    for (char *c = path + 9; *c; ++c) if (*c == '/') *c = '_';
    return open(path, oflag, 0600);
}

int my_shm_unlink(const char *name) {
    char path[256];
    const char *p = (name[0] == '/') ? name + 1 : name;
    snprintf(path, sizeof(path), "/tmp/shm_%s", p);
    for (char *c = path + 9; *c; ++c) if (*c == '/') *c = '_';
    return unlink(path);
}

typedef struct { const void *replacement; const void *replacee; } interpose_t;
__attribute__((used)) static const interpose_t interposers[] __attribute__((section("__DATA,__interpose"))) = {
    { (const void *)my_shm_open, (const void *)shm_open },
    { (const void *)my_shm_unlink, (const void *)shm_unlink },
};
