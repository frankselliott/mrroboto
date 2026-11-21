/* Ruko Companion native launcher.
 *
 * macOS attributes Bluetooth and microphone permission to the *code identity* of the
 * responsible process. A bundle whose main executable is a shell script exec()s into
 * /bin/bash, an Apple platform binary, so permission prompts never appear for it. This tiny
 * program keeps the bundle's own identity alive: it spawns launch.sh as a child, forwards
 * SIGTERM/SIGINT, and exits with the child's status.
 */
#include <libgen.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

int main(int argc, char **argv) {
    char exe[4096];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0) return 1;
    char dir[4096];
    strncpy(dir, dirname(exe), sizeof(dir) - 1);
    char script[4200];
    snprintf(script, sizeof(script), "%s/launch.sh", dir);

    char **args = calloc((size_t)argc + 3, sizeof(char *));
    args[0] = "/bin/bash";
    args[1] = script;
    for (int i = 1; i < argc; i++) args[i + 1] = argv[i];

    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    if (posix_spawn(&child, "/bin/bash", NULL, NULL, args, environ) != 0) {
        perror("posix_spawn");
        return 1;
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) { /* retry on EINTR */ }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 0;
}
