package org.ironman.framework.util;

import android.os.IBinder;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

import java.lang.reflect.Type;

public class GsonUtil {

    private final static Gson gson = new GsonBuilder()
            .registerTypeAdapter(IBinder.class, new BinderSerializer())
            .create();

    public static String toJson(Object src) {
        return gson.toJson(src);
    }

    public String toJson(Object src, Type typeOfSrc) {
        return gson.toJson(src, typeOfSrc);
    }

    private static class BinderSerializer implements JsonSerializer<IBinder> {

        public JsonElement serialize(
                IBinder binder,
                Type type,
                JsonSerializationContext jsonSerializationContext) {
            return new JsonPrimitive(binder.toString());
        }
    }
}
