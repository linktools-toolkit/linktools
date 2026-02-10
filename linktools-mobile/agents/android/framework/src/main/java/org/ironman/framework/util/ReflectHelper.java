package org.ironman.framework.util;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.NoSuchElementException;

/**
 * Created by HuJi on 2017/12/29.
 * <p>easy to use java reflection</p>
 * <pre>type ccc = ReflectHelper.getDefault().invoke("aaa", "bbb");</pre>
 */

@SuppressWarnings({"unused", "WeakerAccess", "SameParameterValue"})
public class ReflectHelper {

    private static final ReflectHelper DEFAULT = new ReflectHelper();

    public static ReflectHelper getDefault() {
        return DEFAULT;
    }

    public static ReflectHelper get(ClassLoader classLoader) {
        return classLoader == null ? getDefault() : new ReflectHelper(classLoader);
    }

    private final ClassLoader mClassLoader;

    private ReflectHelper() {
        mClassLoader = ReflectHelper.class.getClassLoader();
    }

    private ReflectHelper(ClassLoader classLoader) {
        mClassLoader = classLoader;
    }

    public ClassLoader getClassLoader() {
        return mClassLoader;
    }

    public Class<?> loadClass(String className)
            throws ClassNotFoundException {
        switch (className) {
            case "boolean": return boolean.class;
            case "byte": return byte.class;
            case "char": return char.class;
            case "short": return short.class;
            case "int": return int.class;
            case "long": return long.class;
            case "float": return float.class;
            case "double": return double.class;
            case "void": return void.class;
        }
        return Class.forName(className, true, getClassLoader());
    }

    public Class<?> getWrapperClass(Class<?> clazz) {
        switch (clazz.getName()) {
            case "boolean": return Boolean.class;
            case "byte": return Byte.class;
            case "char": return Character.class;
            case "short": return Short.class;
            case "int": return Integer.class;
            case "long": return Long.class;
            case "float": return Float.class;
            case "double": return Double.class;
            case "void": return Void.class;
        }
        return clazz;
    }

    public Method findMatchedMethod(String className, String methodName, Object... args)
            throws ClassNotFoundException {
        return findMatchedMethod(loadClass(className), methodName, args);
    }

    public Method findMatchedMethod(Class<?> clazz, String methodName, Object... args) {
        Class<?> tmp = clazz;
        for (; tmp != null; tmp = tmp.getSuperclass()) {
            Method[] methods = tmp.getDeclaredMethods();
            for (Method method : methods) {
                if (methodName.equals(method.getName()) && matchParameters(method, args)) {
                    return method;
                }
            }
        }
        throw new NoSuchElementException(clazz.getName() + "." + methodName + "()");
    }

    public boolean matchParameters(Method method, Object... params) {
        return matchParameters(method.getParameterTypes(), params);
    }

    public boolean matchParameters(Class<?>[] parameterTypes, Object... params) {
        if (parameterTypes.length != params.length) {
            return false;
        }
        for (int i = 0; i < parameterTypes.length; i++) {
            if (parameterTypes[i].isPrimitive()) {
                if (params[i] == null || !getWrapperClass(parameterTypes[i]).equals(params[i].getClass())) {
                    return false;
                }
            } else {
                if (params[i] != null && !parameterTypes[i].isAssignableFrom(params[i].getClass())) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean matchParameters(Method method, Class<?>... types) {
        return matchParameters(method.getParameterTypes(), types);
    }

    public boolean matchParameters(Class<?>[] parameterTypes, Class<?>... types) {
        if (parameterTypes.length != types.length) {
            return false;
        }
        for (int i = 0; i < parameterTypes.length; i++) {
            if (parameterTypes[i].isAssignableFrom(types[i])) {
                return false;
            }
        }
        return true;
    }

    public Method getMethod(String className, String methodName, Object... parameterTypes)
            throws ClassNotFoundException, NoSuchMethodException {
        return getMethod(loadClass(className), methodName, getClassType(parameterTypes));
    }

    public Method getMethod(String className, String methodName, Class<?>... parameterTypes)
            throws ClassNotFoundException, NoSuchMethodException {
        return getMethod(loadClass(className), methodName, parameterTypes);
    }

    public Method getMethod(Class<?> clazz, String methodName, Object... parameterTypes)
            throws ClassNotFoundException, NoSuchMethodException {
        return getMethod(clazz, methodName, getClassType(parameterTypes));
    }

    public Method getMethod(Class<?> clazz, String methodName, Class<?>... parameterTypes)
            throws NoSuchMethodException {
        NoSuchMethodException exception = null;
        for (; clazz != null; clazz = clazz.getSuperclass()) {
            try {
                Method method = clazz.getDeclaredMethod(methodName, parameterTypes);
                if (!method.isAccessible()) {
                    method.setAccessible(true);
                }
                return method;
            } catch (NoSuchMethodException e) {
                if (exception == null) {
                    exception = e;
                }
            }
        }
        //noinspection ConstantConditions
        throw exception;
    }

    @SuppressWarnings("unchecked")
    public <T> T invoke(String className, String methodName)
            throws InvocationTargetException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {
        return (T) getMethod(loadClass(className), methodName).invoke(null);
    }

    @SuppressWarnings("unchecked")
    public <T> T invoke(Class<?> clazz, String methodName)
            throws InvocationTargetException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {
        return (T) getMethod(clazz, methodName).invoke(null);
    }

    @SuppressWarnings("unchecked")
    public <T> T invoke(Object object, String methodName)
            throws InvocationTargetException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {
        return (T) getMethod(object.getClass(), methodName).invoke(object);
    }

    @SuppressWarnings("unchecked")
    public <T> T  invoke(String className, String methodName, Object... parameterTypesAndParameters)
            throws InvocationTargetException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {
        int length = parameterTypesAndParameters.length;
        Class<?>[] parameterTypes = getClassType(parameterTypesAndParameters, 0, length >> 1);
        Object[] parameters = getParameters(parameterTypesAndParameters, length >> 1, length >> 1);
        return (T) getMethod(loadClass(className), methodName, parameterTypes).invoke(null, parameters);
    }

    @SuppressWarnings("unchecked")
    public <T> T invoke(Class<?> clazz, String methodName, Object... parameterTypesAndParameters)
            throws InvocationTargetException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {
        int length = parameterTypesAndParameters.length;
        Class<?>[] parameterTypes = getClassType(parameterTypesAndParameters, 0, length >> 1);
        Object[] parameters = getParameters(parameterTypesAndParameters, length >> 1, length >> 1);
        return (T) getMethod(clazz, methodName, parameterTypes).invoke(null, parameters);
    }

    @SuppressWarnings("unchecked")
    public <T> T invoke(Object object, String methodName, Object... parameterTypesAndParameters)
            throws InvocationTargetException, IllegalAccessException, ClassNotFoundException, NoSuchMethodException {
        int length = parameterTypesAndParameters.length;
        Class<?>[] parameterTypes = getClassType(parameterTypesAndParameters, 0, length >> 1);
        Object[] parameters = getParameters(parameterTypesAndParameters, length >> 1, length >> 1);
        return (T) getMethod(object.getClass(), methodName, parameterTypes).invoke(object, parameters);
    }

    public Constructor<?> getConstructor(String className, Object... parameterTypes)
            throws ClassNotFoundException, NoSuchMethodException {
        return getConstructor(loadClass(className), getClassType(parameterTypes));
    }

    public Constructor<?> getConstructor(String className, Class<?>... parameterTypes)
            throws ClassNotFoundException, NoSuchMethodException {
        return getConstructor(loadClass(className), parameterTypes);
    }

    public Constructor<?> getConstructor(Class<?> clazz, Object... parameterTypes)
            throws ClassNotFoundException, NoSuchMethodException {
        return getConstructor(clazz, getClassType(parameterTypes));
    }

    public Constructor<?> getConstructor(Class<?> clazz, Class<?>... parameterTypes)
            throws NoSuchMethodException {
        Constructor<?> constructor = clazz.getDeclaredConstructor(parameterTypes);
        if (!constructor.isAccessible()) {
            constructor.setAccessible(true);
        }
        return constructor;
    }

    @SuppressWarnings("unchecked")
    public <T> T newInstance(String className)
            throws ClassNotFoundException, NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        return (T) getConstructor(loadClass(className)).newInstance();
    }

    @SuppressWarnings("unchecked")
    public <T> T newInstance(Class<?> clazz)
            throws NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        return (T) getConstructor(clazz).newInstance();
    }

    @SuppressWarnings("unchecked")
    public <T> T newInstance(String className, Object... parameterTypesAndParameters)
            throws ClassNotFoundException, NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        int length = parameterTypesAndParameters.length;
        Class<?>[] parameterTypes = getClassType(parameterTypesAndParameters, 0, length >> 1);
        Object[] parameters = getParameters(parameterTypesAndParameters, length >> 1, length >> 1);
        return (T) getConstructor(loadClass(className), parameterTypes).newInstance(parameters);
    }

    @SuppressWarnings("unchecked")
    public <T> T newInstance(Class clazz, Object... parameterTypesAndParameters)
            throws ClassNotFoundException, NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        int length = parameterTypesAndParameters.length;
        Class<?>[] parameterTypes = getClassType(parameterTypesAndParameters, 0, length >> 1);
        Object[] parameters = getParameters(parameterTypesAndParameters, length >> 1, length >> 1);
        return (T) getConstructor(clazz, parameterTypes).newInstance(parameters);
    }

    private Class<?>[] getClassType(Object... parameterTypes)
            throws ClassNotFoundException {
        Class<?>[] type = new Class[parameterTypes != null ? parameterTypes.length : 0];
        for (int i = 0; i < type.length; i++) {
            //noinspection StatementWithEmptyBody
            if (parameterTypes == null) {
                // ignore
            } else if (parameterTypes[i] instanceof Class<?>) {
                type[i] = (Class<?>) parameterTypes[i];
            } else if (parameterTypes[i] instanceof String) {
                type[i] = loadClass((String) parameterTypes[i]);
            } else {
                type[i] = parameterTypes[i].getClass();
            }
        }
        return type;
    }

    private Class<?>[] getClassType(Object[] parameterTypes, int offset, int length)
            throws ClassNotFoundException {
        Class<?>[] type = new Class[length];
        for (int i = offset; i < offset + length; i++) {
            if (parameterTypes[i] instanceof Class<?>) {
                type[i] = (Class<?>) parameterTypes[i];
            } else if (parameterTypes[i] instanceof String) {
                type[i] = loadClass((String) parameterTypes[i]);
            } else {
                type[i] = parameterTypes[i].getClass();
            }
        }
        return type;
    }

    private Object[] getParameters(Object[] parameters, int offset, int length) {
        Object[] params = new Object[length];
        System.arraycopy(parameters, offset, params, 0, length);
        return params;
    }

    public Field getField(Object object, String fieldName)
            throws NoSuchFieldException {
        return getField(object.getClass(), fieldName);
    }

    public Field getField(Class<?> clazz, String fieldName)
            throws NoSuchFieldException {
        NoSuchFieldException exception = null;
        for (; clazz != null; clazz = clazz.getSuperclass()) {
            try {
                Field field = clazz.getDeclaredField(fieldName);
                if (!field.isAccessible()) {
                    field.setAccessible(true);
                }
                return field;
            } catch (NoSuchFieldException e) {
                if (exception == null) {
                    exception = e;
                }
            }
        }
        //noinspection ConstantConditions
        throw exception;
    }

    @SuppressWarnings("unchecked")
    public <T> T get(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return (T) getField(loadClass(className), fieldName).get(null);
    }

    @SuppressWarnings("unchecked")
    public <T> T get(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return (T) getField(clazz, fieldName).get(null);
    }

    @SuppressWarnings("unchecked")
    public <T> T get(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return (T) getField(object.getClass(), fieldName).get(object);
    }

    public void set(String className, String fieldName, Object value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).set(null, value);
    }

    public void set(Class<?> clazz, String fieldName, Object value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).set(null, value);
    }

    public void set(Object object, String fieldName, Object value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).set(object, value);
    }

    public boolean getBoolean(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getBoolean(null);
    }

    public boolean getBoolean(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getBoolean(null);
    }

    public boolean getBoolean(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getBoolean(object);
    }

    public void setBoolean(String className, String fieldName, boolean value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setBoolean(null, value);
    }

    public void setBoolean(Class<?> clazz, String fieldName, boolean value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setBoolean(null, value);
    }

    public void setBoolean(Object object, String fieldName, boolean value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setBoolean(object, value);
    }

    public byte getByte(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getByte(null);
    }

    public byte getByte(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getByte(null);
    }

    public byte getByte(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getByte(object);
    }

    public void setBoolean(String className, String fieldName, byte value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setByte(null, value);
    }

    public void setBoolean(Class<?> clazz, String fieldName, byte value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setByte(null, value);
    }

    public void setByte(Object object, String fieldName, byte value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setByte(object, value);
    }

    public char getChar(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getChar(null);
    }

    public char getChar(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getChar(null);
    }

    public char getChar(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getChar(object);
    }

    public void setBoolean(String className, String fieldName, char value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setChar(null, value);
    }

    public void setBoolean(Class<?> clazz, String fieldName, char value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setChar(null, value);
    }

    public void setChar(Object object, String fieldName, char value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setChar(object, value);
    }

    public short getShort(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getShort(null);
    }

    public short getShort(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getShort(null);
    }

    public short getShort(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getShort(object);
    }

    public void setShort(String className, String fieldName, short value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setShort(null, value);
    }

    public void setShort(Class<?> clazz, String fieldName, short value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setShort(null, value);
    }

    public void setShort(Object object, String fieldName, short value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setShort(object, value);
    }

    public int getInt(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getInt(null);
    }

    public int getInt(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getInt(null);
    }

    public int getInt(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getInt(object);
    }

    public void setInt(String className, String fieldName, int value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setInt(null, value);
    }

    public void setInt(Class<?> clazz, String fieldName, int value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setInt(null, value);
    }

    public void setInt(Object object, String fieldName, int value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setInt(object, value);
    }

    public long getLong(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getLong(null);
    }

    public long getLong(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getLong(null);
    }

    public long getLong(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getLong(object);
    }

    public void setLong(String className, String fieldName, long value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setLong(null, value);
    }

    public void setLong(Class<?> clazz, String fieldName, long value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setLong(null, value);
    }

    public void setLong(Object object, String fieldName, long value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setLong(object, value);
    }

    public float getFloat(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getFloat(null);
    }

    public float getFloat(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getFloat(null);
    }

    public float getFloat(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getFloat(object);
    }

    public void setFloat(String className, String fieldName, float value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setFloat(null, value);
    }

    public void setFloat(Class<?> clazz, String fieldName, float value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setFloat(null, value);
    }

    public void setFloat(Object object, String fieldName, float value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setFloat(object, value);
    }

    public double getDouble(String className, String fieldName)
            throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException {
        return getField(loadClass(className), fieldName).getDouble(null);
    }

    public double getDouble(Class<?> clazz, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(clazz, fieldName).getDouble(null);
    }

    public double getDouble(Object object, String fieldName)
            throws NoSuchFieldException, IllegalAccessException {
        return getField(object.getClass(), fieldName).getDouble(object);
    }

    public void setDouble(String className, String fieldName, double value)
            throws NoSuchFieldException, IllegalAccessException, ClassNotFoundException {
        getField(loadClass(className), fieldName).setDouble(null, value);
    }

    public void setDouble(Class<?> clazz, String fieldName, double value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(clazz, fieldName).setDouble(null, value);
    }

    public void setDouble(Object object, String fieldName, double value)
            throws NoSuchFieldException, IllegalAccessException {
        getField(object.getClass(), fieldName).setDouble(object, value);
    }
}
