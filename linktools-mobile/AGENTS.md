# AGENTS.md (linktools-mobile)

Architecture guidance for the mobile-tools sub-package. Shared concerns (monorepo
structure, `manage.py`, config system, code style) live in the
[repo-root AGENTS.md](../AGENTS.md).

## Mobile Sub-package (`linktools-mobile/src/linktools/mobile/`)

- **`android/`** — `_adb.py` (ADB wrapper with multi-device selection), `_scrcpy.py` (scrcpy bridge), `_types.py` (Android-specific types)
- **`ios/`** — `_ios.py` (go-ios wrapper), `_ipa.py` (IPA parser), `_sib.py`, `_types.py`
- **`frida/`** — Frida integration: `_app.py` (`FridaApplication`, `FridaSession`, `FridaScript`, `FridaReactor`), `_server.py` (`FridaServer`, `FridaAndroidServer`, `FridaIOSServer`), `_script.py` (`FridaUserScript`, `FridaEvalCode`, `FridaScriptFile`)

## Frida TypeScript Agents (`agents/frida/`)

TypeScript source for the built-in Frida scripts. The compiled output (`frida.js`,
`frida-*.js`) is committed to `src/linktools/assets/` as a build artifact. Key library
in `lib/java.ts` provides `JavaHelper` with `hookMethod`, `hookMethods`,
`hookAllMethods`, `bypassSslPinning`, etc.

```bash
cd agents/frida
npm install
npm run build
```

## Android APK Agent (`agents/android/`)

Gradle/Android project that builds `android-tools.apk`. The built APK is committed to
`src/linktools/assets/android-tools.*` as a build artifact.

```bash
cd agents/android
./gradlew --no-daemon :tools:buildTools
```
