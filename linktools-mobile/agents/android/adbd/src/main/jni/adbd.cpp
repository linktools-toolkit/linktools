//
// Created by hu on 18-5-22.
//
#include "jni.h"
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>

extern "C" int adb_trace_mask;
extern "C" int adb_main(int is_daemon, int server_port);

extern "C" jint Java_org_ironman_adbd_Adbd_nativeRun(
    JNIEnv* env, jclass cls, jint daemon, jint port, jobjectArray envs, jint traceMask) {
    pid_t pid = fork();
    if (pid == 0) {
        jsize size = env->GetArrayLength(envs);
        for (int i = 0; i < size; i++) {
            auto item = (jstring) env->GetObjectArrayElement(envs, i);
            const char *string = env->GetStringUTFChars(item, nullptr);
            putenv((char*) string);
//            env->ReleaseStringUTFChars(obj, string);
        }
        adb_trace_mask = traceMask;
        exit(adb_main(daemon, port));
    }
    return pid;
}

extern "C" jboolean Java_org_ironman_adbd_Adbd_nativeIsRunning(JNIEnv* env, jclass cls, jint pid) {
    return waitpid(pid, nullptr, WNOHANG) == 0;
}
