# Copyright (C) 2009 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

PROJECT_ADBD_PATH := $(call my-dir)

LOCAL_PATH := ${PROJECT_ADBD_PATH}
include $(call all-makefiles-under,$(LOCAL_PATH))

LOCAL_PATH := ${PROJECT_ADBD_PATH}
include $(CLEAR_VARS)
LOCAL_MODULE := adbd
LOCAL_SRC_FILES := adbd.cpp
LOCAL_STATIC_LIBRARIES := adbd_static
LOCAL_C_INCLUDES += $(LOCAL_PATH)/adbd/ $(LOCAL_PATH)/adbd/include/
LOCAL_LDLIBS := -llog
include $(BUILD_SHARED_LIBRARY)

LOCAL_PATH := ${PROJECT_ADBD_PATH}
include $(CLEAR_VARS)
LOCAL_MODULE := adbd-pie
LOCAL_STATIC_LIBRARIES := adbd_static
LOCAL_C_INCLUDES += $(LOCAL_PATH)/adbd/ $(LOCAL_PATH)/adbd/include/
LOCAL_LDLIBS := -llog
LOCAL_CFLAGS += -pie -fPIE
LOCAL_LDFLAGS += -pie -fPIE
include $(BUILD_EXECUTABLE)

LOCAL_PATH := ${PROJECT_ADBD_PATH}
include $(CLEAR_VARS)
LOCAL_MODULE := adbd-nopie
LOCAL_STATIC_LIBRARIES := adbd_static
LOCAL_C_INCLUDES += $(LOCAL_PATH)/adbd/ $(LOCAL_PATH)/adbd/include/
LOCAL_LDLIBS := -llog
include $(BUILD_EXECUTABLE)