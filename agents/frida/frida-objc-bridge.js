import ObjC from "frida-objc-bridge";

Object.defineProperty(globalThis, 'ObjC', { value: ObjC });
