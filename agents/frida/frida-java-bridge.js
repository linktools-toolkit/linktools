import Java from "frida-java-bridge";

Object.defineProperty(globalThis, 'Java', { value: Java });
